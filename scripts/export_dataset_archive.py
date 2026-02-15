#!/usr/bin/env python3
"""
Dataset Archive Export for Local GPU Training

Downloads all automotive training data from S3 and packages it into a
self-contained tar.gz (or zip) archive for transport to a local GPU setup.

Packages:
  - Raw source code from all 24 automotive domains (S3 source bucket)
  - Processed train.jsonl / val.jsonl splits (S3 output bucket)
  - Optional teacher outputs (local)
  - config.yaml and a manifest.json with provenance metadata

Usage:
    # Full archive with all data
    python scripts/export_dataset_archive.py

    # Only processed splits (fast, ~1.1GB)
    python scripts/export_dataset_archive.py --skip-raw

    # Include teacher outputs
    python scripts/export_dataset_archive.py --include-teacher-outputs

    # Dry run — preview without downloading
    python scripts/export_dataset_archive.py --dry-run

    # Custom output path
    python scripts/export_dataset_archive.py --output-path /mnt/nas/dataset.tar.gz

    # Upload archive to S3 (ideal for EC2 workflow)
    python scripts/export_dataset_archive.py --upload-to-s3 s3://my-bucket/archives/

Author: Sriram Acharya
Organization: Excelfore
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import yaml
from botocore.exceptions import ClientError, NoCredentialsError
from tqdm import tqdm

# Load environment variables (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import DATA_SOURCES and CODE_EXTENSIONS from the existing pipeline
sys.path.insert(0, str(Path(__file__).parent))
from prepare_automotive_data import AutomotiveDataPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR_NAME = "granite-8b-automotive-dataset"


def format_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def count_lines(file_path: Path) -> int:
    """Count lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


class DatasetArchiver:
    """
    Download and package all automotive training data from S3
    into a self-contained archive for local GPU training.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.source_bucket = args.source_bucket
        self.output_bucket = args.output_bucket
        self.splits_prefix = args.splits_prefix
        self.region = args.region

        # Staging directory
        if args.staging_dir:
            self.staging_dir = Path(args.staging_dir)
            self.staging_dir.mkdir(parents=True, exist_ok=True)
            self._owns_staging = False
        else:
            self.staging_dir = Path(tempfile.mkdtemp(prefix='granite-archive-'))
            self._owns_staging = True

        self.archive_root = self.staging_dir / ARCHIVE_DIR_NAME
        self.s3_client: Optional[boto3.client] = None
        self.use_pigz = False
        self.manifest_data: Dict = {}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_aws_access(self) -> bool:
        """Validate AWS credentials and bucket access."""
        print("\n" + "=" * 70)
        print("[Archiver] VALIDATING AWS ACCESS")
        print("=" * 70)

        # STS identity
        try:
            sts = boto3.client('sts', region_name=self.region)
            identity = sts.get_caller_identity()
            print(f"[Archiver] AWS Account: {identity['Account']}")
        except NoCredentialsError:
            print("[Archiver] ERROR: No AWS credentials found.")
            print("  Run: aws configure")
            print("  Or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in .env")
            return False
        except Exception as e:
            print(f"[Archiver] ERROR: AWS STS check failed — {e}")
            return False

        self.s3_client = boto3.client('s3', region_name=self.region)

        # Check source bucket (raw data)
        if not self.args.skip_raw:
            try:
                self.s3_client.head_bucket(Bucket=self.source_bucket)
                print(f"[Archiver] [OK] Source bucket: {self.source_bucket}")
            except ClientError as e:
                print(f"[Archiver] [FAIL] Source bucket ({self.source_bucket}) — {e}")
                return False

        # Check output bucket (processed splits)
        if not self.args.skip_processed:
            try:
                self.s3_client.head_bucket(Bucket=self.output_bucket)
                print(f"[Archiver] [OK] Output bucket: {self.output_bucket}")
            except ClientError as e:
                print(f"[Archiver] [FAIL] Output bucket ({self.output_bucket}) — {e}")
                return False

        return True

    def check_pigz_available(self):
        """Check if pigz (parallel gzip) is available for faster compression."""
        try:
            result = subprocess.run(
                ['which', 'pigz'], capture_output=True, text=True
            )
            self.use_pigz = result.returncode == 0
        except Exception:
            self.use_pigz = False

        if self.use_pigz:
            print("[Archiver] pigz found — will use parallel compression")
        else:
            print("[Archiver] pigz not found — using standard gzip")

    # ------------------------------------------------------------------
    # Staging
    # ------------------------------------------------------------------

    def setup_staging_dir(self):
        """Create the staging directory structure."""
        print(f"[Archiver] Staging directory: {self.staging_dir}")

        dirs = [self.archive_root / "raw", self.archive_root / "processed"]
        if self.args.include_teacher_outputs:
            dirs.append(self.archive_root / "teacher_outputs")

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Downloads
    # ------------------------------------------------------------------

    def _list_s3_objects(self, bucket: str, prefix: str = "") -> List[Dict]:
        """List all S3 objects in the bucket (no extension filtering)."""
        paginator = self.s3_client.get_paginator('list_objects_v2')

        objects = []
        total_size = 0
        page_config = {'Bucket': bucket}
        if prefix:
            page_config['Prefix'] = prefix

        for page in paginator.paginate(**page_config):
            for obj in page.get('Contents', []):
                key = obj['Key']
                # Skip directory markers (zero-byte keys ending in /)
                if key.endswith('/') and obj['Size'] == 0:
                    continue
                objects.append({'key': key, 'size': obj['Size']})
                total_size += obj['Size']

        return objects

    def _download_s3_file(
        self, bucket: str, key: str, local_path: Path
    ) -> Tuple[bool, str]:
        """Download a single file from S3. Returns (success, key)."""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(bucket, key, str(local_path))
            return True, key
        except Exception as e:
            return False, f"{key}: {e}"

    def download_raw_data(self) -> bool:
        """Download raw source code from S3 using boto3 parallel downloads."""
        if self.args.skip_raw:
            print("[Archiver] Skipping raw data (--skip-raw)")
            return True

        workers = self.args.workers
        local_dir = self.archive_root / "raw"

        # Phase 1: List all objects
        print(f"[Archiver] Listing all objects in s3://{self.source_bucket}/ ...")

        objects = self._list_s3_objects(self.source_bucket)
        total_size = sum(obj['size'] for obj in objects)

        print(f"[Archiver] Found {len(objects):,} files ({format_size(total_size)})")

        if not objects:
            print("[Archiver] WARNING: No matching files found in bucket")
            return True

        if self.args.dry_run:
            print(f"\n[Archiver] DRY RUN — would download {len(objects):,} files "
                  f"({format_size(total_size)}) with {workers} workers")
            # Show per-domain breakdown
            domain_counts = {}
            for obj in objects:
                key = obj['key']
                matched = False
                for source in AutomotiveDataPipeline.DATA_SOURCES:
                    if key.startswith(source['prefix']):
                        domain_counts[source['domain']] = domain_counts.get(source['domain'], 0) + 1
                        matched = True
                        break
                if not matched:
                    domain_counts['other'] = domain_counts.get('other', 0) + 1

            print(f"\n[Archiver] Per-domain file counts:")
            for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
                print(f"  {domain:25s} {count:>8,}")
            return True

        # Phase 2: Parallel download
        print(f"[Archiver] Downloading with {workers} parallel workers...")
        print("-" * 70)

        failed = []
        downloaded_bytes = 0
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for obj in objects:
                key = obj['key']
                local_path = local_dir / key
                future = executor.submit(
                    self._download_s3_file, self.source_bucket, key, local_path
                )
                futures[future] = obj

            with tqdm(total=len(objects), desc="Downloading", unit="files") as pbar:
                for future in as_completed(futures):
                    obj = futures[future]
                    success, msg = future.result()
                    if success:
                        with lock:
                            downloaded_bytes += obj['size']
                    else:
                        failed.append(msg)
                    pbar.update(1)
                    # Update postfix with running size
                    pbar.set_postfix_str(format_size(downloaded_bytes), refresh=False)

        print(f"\n[Archiver] Raw data download complete: "
              f"{len(objects) - len(failed):,} files ({format_size(downloaded_bytes)})")

        if failed:
            print(f"[Archiver] WARNING: {len(failed):,} files failed to download")
            for msg in failed[:10]:
                print(f"  - {msg}")
            if len(failed) > 10:
                print(f"  ... and {len(failed) - 10} more")

        return True

    def download_processed_splits(self) -> bool:
        """Download processed train.jsonl and val.jsonl from S3."""
        if self.args.skip_processed:
            print("[Archiver] Skipping processed splits (--skip-processed)")
            return True

        split_files = ['train.jsonl', 'val.jsonl']

        for split_name in split_files:
            s3_key = f"{self.splits_prefix}/{split_name}"

            if self.args.dry_run:
                try:
                    obj = self.s3_client.head_object(
                        Bucket=self.output_bucket, Key=s3_key
                    )
                    size = format_size(obj['ContentLength'])
                    print(f"[Archiver] DRY RUN — would download {split_name} ({size})")
                except ClientError:
                    print(f"[Archiver] DRY RUN — {split_name} NOT FOUND at s3://{self.output_bucket}/{s3_key}")
                continue

            local_path = self.archive_root / "processed" / split_name

            try:
                obj = self.s3_client.head_object(
                    Bucket=self.output_bucket, Key=s3_key
                )
                size = format_size(obj['ContentLength'])
                print(f"[Archiver] Downloading {split_name} ({size}) from s3://{self.output_bucket}/{s3_key}")
                self.s3_client.download_file(
                    self.output_bucket, s3_key, str(local_path)
                )
            except ClientError as e:
                print(f"[Archiver] ERROR: Could not download {split_name} — {e}")
                return False

        if not self.args.dry_run:
            print("[Archiver] Processed splits downloaded")
        return True

    def copy_teacher_outputs(self) -> bool:
        """Copy teacher outputs from local data directory if requested."""
        if not self.args.include_teacher_outputs:
            print("[Archiver] Skipping teacher outputs (use --include-teacher-outputs to include)")
            return True

        source = PROJECT_ROOT / "data" / "teacher_outputs" / "bedrock_outputs.jsonl"

        if self.args.dry_run:
            if source.exists():
                size = format_size(source.stat().st_size)
                print(f"[Archiver] DRY RUN — would copy {source.name} ({size})")
            else:
                print(f"[Archiver] DRY RUN — teacher outputs not found at {source}")
            return True

        if not source.exists():
            print(f"[Archiver] WARNING: Teacher outputs not found at {source}")
            print("[Archiver] Skipping teacher outputs (file does not exist)")
            return True

        dest = self.archive_root / "teacher_outputs" / source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        size = format_size(dest.stat().st_size)
        print(f"[Archiver] Copied {source.name} ({size})")
        return True

    def copy_config(self):
        """Copy config.yaml into the archive."""
        source = PROJECT_ROOT / "config.yaml"

        if self.args.dry_run:
            if source.exists():
                print(f"[Archiver] DRY RUN — would copy config.yaml")
            else:
                print(f"[Archiver] DRY RUN — config.yaml not found")
            return

        if source.exists():
            shutil.copy2(source, self.archive_root / "config.yaml")
            print("[Archiver] Copied config.yaml")
        else:
            print(f"[Archiver] WARNING: config.yaml not found at {source}")

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def collect_manifest_data(self) -> Dict:
        """Walk the staged directories and collect metadata for the manifest."""
        raw_stats = {}
        raw_total_files = 0
        raw_total_size = 0

        # Per-domain stats from raw/
        for source in AutomotiveDataPipeline.DATA_SOURCES:
            prefix = source['prefix'].rstrip('/')
            domain = source['domain']
            domain_dir = self.archive_root / "raw" / prefix

            if not domain_dir.exists():
                raw_stats[domain] = {
                    "prefix": source['prefix'],
                    "file_count": 0,
                    "total_size_bytes": 0,
                }
                continue

            file_count = 0
            total_size = 0
            for ext in AutomotiveDataPipeline.CODE_EXTENSIONS:
                for f in domain_dir.rglob(f'*{ext}'):
                    file_count += 1
                    total_size += f.stat().st_size

            raw_stats[domain] = {
                "prefix": source['prefix'],
                "file_count": file_count,
                "total_size_bytes": total_size,
            }
            raw_total_files += file_count
            raw_total_size += total_size

        # Processed splits stats
        split_stats = {}
        processed_total_lines = 0
        processed_total_size = 0

        for split_name in ['train.jsonl', 'val.jsonl']:
            split_path = self.archive_root / "processed" / split_name
            if split_path.exists():
                size = split_path.stat().st_size
                lines = count_lines(split_path)
                split_stats[split_name] = {
                    "line_count": lines,
                    "size_bytes": size,
                }
                processed_total_lines += lines
                processed_total_size += size
            else:
                split_stats[split_name] = {"line_count": 0, "size_bytes": 0}

        # Teacher outputs
        teacher_included = False
        teacher_files = []
        teacher_dir = self.archive_root / "teacher_outputs"
        if teacher_dir.exists():
            for f in teacher_dir.iterdir():
                if f.is_file():
                    teacher_included = True
                    teacher_files.append({
                        "name": f.name,
                        "line_count": count_lines(f),
                        "size_bytes": f.stat().st_size,
                    })

        # Project version from config
        project_version = "unknown"
        project_name = "granite-8b-avb-tsn-finetuning"
        config_path = self.archive_root / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f)
                project_version = cfg.get('project', {}).get('version', 'unknown')
                project_name = cfg.get('project', {}).get('name', project_name)
            except Exception:
                pass

        return {
            "creation_timestamp": datetime.now(timezone.utc).isoformat(),
            "source_buckets": {
                "raw_data": f"s3://{self.source_bucket}/",
                "processed_splits": f"s3://{self.output_bucket}/{self.splits_prefix}/",
            },
            "raw_data": raw_stats,
            "processed_splits": split_stats,
            "teacher_outputs": {
                "included": teacher_included,
                "files": teacher_files,
            },
            "archive": {
                "format": self.args.format,
                "compression_method": (
                    "pigz" if (self.args.format == 'tar.gz' and self.use_pigz)
                    else "gzip" if self.args.format == 'tar.gz'
                    else "deflate"
                ),
                "output_path": str(self.args.output_path),
            },
            "project": {
                "version": project_version,
                "name": project_name,
            },
            "totals": {
                "raw_file_count": raw_total_files,
                "raw_total_size_bytes": raw_total_size,
                "processed_line_count": processed_total_lines,
                "processed_total_size_bytes": processed_total_size,
            },
        }

    def generate_manifest(self):
        """Generate manifest.json with provenance metadata."""
        self.manifest_data = self.collect_manifest_data()

        manifest_path = self.archive_root / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest_data, f, indent=2)

        print(f"[Archiver] Manifest written to {manifest_path}")

        # Print domain summary
        totals = self.manifest_data['totals']
        print(f"[Archiver] Raw data: {totals['raw_file_count']:,} files ({format_size(totals['raw_total_size_bytes'])})")
        print(f"[Archiver] Processed: {totals['processed_line_count']:,} examples ({format_size(totals['processed_total_size_bytes'])})")

        if self.manifest_data['teacher_outputs']['included']:
            for tf in self.manifest_data['teacher_outputs']['files']:
                print(f"[Archiver] Teacher: {tf['name']} ({tf['line_count']:,} examples)")

    # ------------------------------------------------------------------
    # Archive creation
    # ------------------------------------------------------------------

    def create_archive(self) -> Path:
        """Create the archive in the requested format."""
        output_path = Path(self.args.output_path)

        # Ensure correct extension
        if self.args.format == 'tar.gz' and not str(output_path).endswith('.tar.gz'):
            output_path = output_path.with_suffix('.tar.gz')
        elif self.args.format == 'zip' and not str(output_path).endswith('.zip'):
            output_path = output_path.with_suffix('.zip')

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.args.format == 'tar.gz':
            return self._create_tar_gz(output_path)
        else:
            return self._create_zip(output_path)

    def _create_tar_gz(self, output_path: Path) -> Path:
        """Create tar.gz archive using subprocess for performance."""
        print(f"[Archiver] Creating tar.gz archive at {output_path}")
        start = time.time()

        try:
            if self.use_pigz:
                cmd = (
                    f"tar cf - -C '{self.staging_dir}' '{ARCHIVE_DIR_NAME}' "
                    f"| pigz > '{output_path}'"
                )
                subprocess.run(cmd, shell=True, check=True)
            else:
                cmd = [
                    'tar', 'czf', str(output_path),
                    '-C', str(self.staging_dir),
                    ARCHIVE_DIR_NAME,
                ]
                subprocess.run(cmd, check=True)

            elapsed = time.time() - start
            size = format_size(output_path.stat().st_size)
            print(f"[Archiver] Archive created: {size} in {elapsed:.1f}s")
            return output_path

        except subprocess.CalledProcessError as e:
            print(f"[Archiver] ERROR: tar failed — {e}")
            sys.exit(1)

    def _create_zip(self, output_path: Path) -> Path:
        """Create zip archive using Python zipfile."""
        print(f"[Archiver] Creating zip archive at {output_path}")
        start = time.time()

        all_files = list(self.archive_root.rglob('*'))
        all_files = [f for f in all_files if f.is_file()]

        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in tqdm(all_files, desc="Compressing", leave=False):
                arcname = str(file_path.relative_to(self.staging_dir))
                zf.write(file_path, arcname)

        elapsed = time.time() - start
        size = format_size(output_path.stat().st_size)
        print(f"[Archiver] Archive created: {size} in {elapsed:.1f}s")
        return output_path

    # ------------------------------------------------------------------
    # S3 upload
    # ------------------------------------------------------------------

    def upload_archive_to_s3(self, archive_path: Path) -> Optional[str]:
        """Upload the final archive to S3 if --upload-to-s3 is set."""
        upload_uri = self.args.upload_to_s3
        if not upload_uri:
            return None

        # Parse s3://bucket/key
        if not upload_uri.startswith('s3://'):
            print(f"[Archiver] ERROR: --upload-to-s3 must be an s3:// URI")
            return None

        parts = upload_uri[5:].split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''

        # If key is empty or ends with /, treat as prefix and append filename
        if not key or key.endswith('/'):
            key = key + archive_path.name

        if self.args.dry_run:
            print(f"[Archiver] DRY RUN — would upload archive to s3://{bucket}/{key}")
            return f"s3://{bucket}/{key}"

        print(f"\n[Archiver] Uploading archive to s3://{bucket}/{key} ...")
        archive_size = archive_path.stat().st_size
        start = time.time()

        try:
            # Validate bucket access
            self.s3_client.head_bucket(Bucket=bucket)
        except ClientError as e:
            print(f"[Archiver] ERROR: Cannot access bucket '{bucket}' — {e}")
            return None

        # Use multipart upload for large files (>100MB)
        from boto3.s3.transfer import TransferConfig
        config = TransferConfig(
            multipart_threshold=100 * 1024 * 1024,  # 100MB
            max_concurrency=10,
            multipart_chunksize=100 * 1024 * 1024,
        )

        # Progress callback
        uploaded_bytes = 0
        lock = threading.Lock()
        pbar = tqdm(total=archive_size, desc="Uploading", unit='B',
                    unit_scale=True, unit_divisor=1024)

        def upload_progress(bytes_transferred):
            nonlocal uploaded_bytes
            with lock:
                uploaded_bytes += bytes_transferred
                pbar.update(bytes_transferred)

        try:
            self.s3_client.upload_file(
                str(archive_path), bucket, key,
                Config=config,
                Callback=upload_progress,
            )
            pbar.close()

            elapsed = time.time() - start
            s3_uri = f"s3://{bucket}/{key}"
            print(f"[Archiver] Upload complete: {format_size(archive_size)} in {elapsed:.1f}s")
            print(f"[Archiver] Archive location: {s3_uri}")
            return s3_uri

        except Exception as e:
            pbar.close()
            print(f"[Archiver] ERROR: Upload failed — {e}")
            return None

    # ------------------------------------------------------------------
    # Summary and cleanup
    # ------------------------------------------------------------------

    def print_summary(self, elapsed: float, s3_uri: Optional[str] = None):
        """Print final summary."""
        print("\n" + "=" * 70)
        print("[Archiver] EXPORT COMPLETE")
        print("=" * 70)

        if self.args.dry_run:
            print("[Archiver] DRY RUN — no files were downloaded or created")
            print(f"[Archiver] Duration: {elapsed:.1f}s")
            return

        totals = self.manifest_data.get('totals', {})
        output_path = Path(self.args.output_path)

        if not self.args.skip_raw:
            print(f"  Raw source files:    {totals.get('raw_file_count', 0):>10,} files "
                  f"({format_size(totals.get('raw_total_size_bytes', 0))})")
        if not self.args.skip_processed:
            print(f"  Processed examples:  {totals.get('processed_line_count', 0):>10,} lines "
                  f"({format_size(totals.get('processed_total_size_bytes', 0))})")
        if self.manifest_data.get('teacher_outputs', {}).get('included'):
            print(f"  Teacher outputs:     included")

        if output_path.exists():
            archive_size = format_size(output_path.stat().st_size)
            print(f"\n  Archive (local):     {output_path}")
            print(f"  Archive size:        {archive_size}")

        if s3_uri:
            print(f"  Archive (S3):        {s3_uri}")
            print(f"\n  To download:")
            print(f"    aws s3 cp {s3_uri} .")

        print(f"  Duration:            {elapsed / 60:.1f} minutes")

        # Print extraction command
        print(f"\n  To extract:")
        if self.args.format == 'tar.gz':
            print(f"    tar xzf {output_path.name}")
        else:
            print(f"    unzip {output_path.name}")

        print("=" * 70)

    def cleanup(self):
        """Clean up the staging directory."""
        if self.args.dry_run:
            return

        if self.args.keep_staging:
            print(f"[Archiver] Staging directory kept at: {self.staging_dir}")
            return

        if self._owns_staging:
            shutil.rmtree(self.staging_dir, ignore_errors=True)
            print(f"[Archiver] Staging directory cleaned up")

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run(self):
        """Run the full export pipeline."""
        start_time = time.time()

        print("=" * 70)
        print("[Archiver] DATASET ARCHIVE EXPORT")
        print("=" * 70)
        print(f"  Source bucket:   s3://{self.source_bucket}/")
        print(f"  Output bucket:   s3://{self.output_bucket}/{self.splits_prefix}/")
        print(f"  Archive format:  {self.args.format}")
        print(f"  Output path:     {self.args.output_path}")
        print(f"  Workers:         {self.args.workers}")
        print(f"  Skip raw:        {self.args.skip_raw}")
        print(f"  Skip processed:  {self.args.skip_processed}")
        print(f"  Teacher outputs: {self.args.include_teacher_outputs}")
        print(f"  Upload to S3:    {self.args.upload_to_s3 or 'No'}")
        print(f"  Dry run:         {self.args.dry_run}")
        print("=" * 70)

        # Step 1: Validate AWS access
        if not self.validate_aws_access():
            sys.exit(1)

        # Step 2: Check compression tools
        if self.args.format == 'tar.gz':
            self.check_pigz_available()

        # Step 3: Setup staging directory
        if not self.args.dry_run:
            self.setup_staging_dir()

        # Step 4: Download raw data
        print("\n" + "=" * 70)
        print("[Archiver] STEP 1/5: RAW SOURCE DATA")
        print("=" * 70)
        if not self.download_raw_data():
            print("[Archiver] ERROR: Raw data download failed")
            sys.exit(1)

        # Step 5: Download processed splits
        print("\n" + "=" * 70)
        print("[Archiver] STEP 2/5: PROCESSED SPLITS")
        print("=" * 70)
        if not self.download_processed_splits():
            print("[Archiver] ERROR: Processed splits download failed")
            sys.exit(1)

        # Step 6: Copy extras
        print("\n" + "=" * 70)
        print("[Archiver] STEP 3/5: EXTRAS (config, teacher outputs)")
        print("=" * 70)
        self.copy_teacher_outputs()
        self.copy_config()

        if self.args.dry_run:
            if self.args.upload_to_s3:
                self.upload_archive_to_s3(Path(self.args.output_path))
            elapsed = time.time() - start_time
            self.print_summary(elapsed)
            return

        # Step 7: Generate manifest
        print("\n" + "=" * 70)
        print("[Archiver] STEP 4/5: GENERATING MANIFEST")
        print("=" * 70)
        self.generate_manifest()

        # Step 8: Create archive
        print("\n" + "=" * 70)
        step_total = "6" if self.args.upload_to_s3 else "5"
        print(f"[Archiver] STEP 5/{step_total}: CREATING ARCHIVE")
        print("=" * 70)
        archive_path = self.create_archive()

        # Step 9: Upload to S3 (optional)
        s3_uri = None
        if self.args.upload_to_s3:
            print("\n" + "=" * 70)
            print(f"[Archiver] STEP 6/6: UPLOADING TO S3")
            print("=" * 70)
            s3_uri = self.upload_archive_to_s3(archive_path)

        # Summary and cleanup
        elapsed = time.time() - start_time
        self.print_summary(elapsed, s3_uri=s3_uri)
        self.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export automotive dataset to self-contained archive for local GPU training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full archive with all data
  python scripts/export_dataset_archive.py

  # Only processed splits (fast, ~1.1GB)
  python scripts/export_dataset_archive.py --skip-raw

  # Include teacher outputs
  python scripts/export_dataset_archive.py --include-teacher-outputs

  # Dry run — see what would be downloaded
  python scripts/export_dataset_archive.py --dry-run

  # Custom output path and keep staging dir
  python scripts/export_dataset_archive.py --output-path /mnt/nas/dataset.tar.gz --keep-staging

  # Create zip instead of tar.gz
  python scripts/export_dataset_archive.py --format zip
        """
    )
    parser.add_argument(
        '--output-path', type=str,
        default='./granite-8b-automotive-dataset.tar.gz',
        help='Output archive path (default: ./granite-8b-automotive-dataset.tar.gz)'
    )
    parser.add_argument(
        '--format', type=str, choices=['tar.gz', 'zip'],
        default='tar.gz',
        help='Archive format (default: tar.gz)'
    )
    parser.add_argument(
        '--source-bucket', type=str,
        default='granite-8b-unified-automotive-data',
        help='S3 bucket with raw automotive source code'
    )
    parser.add_argument(
        '--output-bucket', type=str,
        default='granite-8b-training-outputs',
        help='S3 bucket with processed train/val splits'
    )
    parser.add_argument(
        '--splits-prefix', type=str,
        default='runs/data/splits',
        help='S3 prefix for processed splits'
    )
    parser.add_argument(
        '--region', type=str,
        default=os.environ.get('AWS_REGION', 'us-east-1'),
        help='AWS region (default: AWS_REGION env or us-east-1)'
    )
    parser.add_argument(
        '--workers', type=int, default=32,
        help='Parallel download threads (default: 32)'
    )
    parser.add_argument(
        '--skip-raw', action='store_true',
        help='Skip raw data download (only package processed splits)'
    )
    parser.add_argument(
        '--skip-processed', action='store_true',
        help='Skip processed splits download'
    )
    parser.add_argument(
        '--include-teacher-outputs', action='store_true',
        help='Include teacher outputs if available locally'
    )
    parser.add_argument(
        '--staging-dir', type=str, default=None,
        help='Staging directory (default: auto temp dir)'
    )
    parser.add_argument(
        '--keep-staging', action='store_true',
        help='Do not clean up staging directory after archive creation'
    )
    parser.add_argument(
        '--upload-to-s3', type=str, default=None,
        metavar='S3_URI',
        help='Upload final archive to S3 (e.g. s3://my-bucket/archives/)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show what would be downloaded without executing'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    archiver = DatasetArchiver(args)
    archiver.run()


if __name__ == "__main__":
    main()
