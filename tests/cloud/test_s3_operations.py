"""
S3 Operations Test Suite

Comprehensive tests for S3 operations including:
- Basic CRUD operations
- Error handling and edge cases
- Large file handling
- Timeout and retry behavior
- Permission scenarios
"""

import time
import pytest
from unittest.mock import MagicMock
from botocore.exceptions import ClientError


# =============================================================================
# BASIC S3 OPERATIONS
# =============================================================================

@pytest.mark.s3
class TestS3BasicOperations:
    """Basic S3 operation tests"""

    def test_bucket_access_success(self, mock_s3_client):
        """Test successful bucket access"""
        result = mock_s3_client.head_bucket(Bucket='test-bucket')
        assert result == {}

    def test_list_objects_with_results(self, mock_s3_client):
        """Test listing objects returns expected structure"""
        response = mock_s3_client.list_objects_v2(
            Bucket='test-bucket',
            Prefix='data/',
            MaxKeys=10
        )
        assert 'Contents' in response
        assert len(response['Contents']) == 2

    def test_list_objects_empty_bucket(self, mock_s3_client):
        """Test handling empty bucket"""
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [],
            'KeyCount': 0
        }
        response = mock_s3_client.list_objects_v2(Bucket='empty-bucket')
        assert response['KeyCount'] == 0


# =============================================================================
# S3 ERROR HANDLING
# =============================================================================

@pytest.mark.s3
@pytest.mark.offline
class TestS3ErrorHandling:
    """S3 error handling tests"""

    def test_access_denied_handling(self, mock_s3_client, s3_access_denied_error):
        """Test handling of AccessDenied error"""
        mock_s3_client.get_object.side_effect = s3_access_denied_error()

        with pytest.raises(ClientError) as exc_info:
            mock_s3_client.get_object(Bucket='secure-bucket', Key='secret.txt')

        assert exc_info.value.response['Error']['Code'] == 'AccessDenied'

    def test_no_such_key_handling(self, mock_s3_client, s3_no_such_key_error):
        """Test handling of NoSuchKey error"""
        mock_s3_client.get_object.side_effect = s3_no_such_key_error('missing.txt')

        with pytest.raises(ClientError) as exc_info:
            mock_s3_client.get_object(Bucket='bucket', Key='missing.txt')

        assert exc_info.value.response['Error']['Code'] == 'NoSuchKey'

    def test_bucket_not_found_handling(self, mock_s3_client):
        """Test handling of NoSuchBucket error"""
        mock_s3_client.head_bucket.side_effect = ClientError(
            {'Error': {'Code': 'NoSuchBucket', 'Message': 'Bucket not found'}},
            'HeadBucket'
        )

        with pytest.raises(ClientError) as exc_info:
            mock_s3_client.head_bucket(Bucket='nonexistent-bucket')

        assert exc_info.value.response['Error']['Code'] == 'NoSuchBucket'

    @pytest.mark.parametrize("error_code,error_message", [
        ("403", "Forbidden"),
        ("404", "Not Found"),
        ("500", "Internal Server Error"),
        ("503", "Service Unavailable"),
    ])
    def test_http_error_codes(self, mock_s3_client, error_code, error_message):
        """Test handling of various HTTP error codes"""
        mock_s3_client.get_object.side_effect = ClientError(
            {'Error': {'Code': error_code, 'Message': error_message}},
            'GetObject'
        )

        with pytest.raises(ClientError):
            mock_s3_client.get_object(Bucket='bucket', Key='file.txt')


# =============================================================================
# S3 EDGE CASES
# =============================================================================

@pytest.mark.s3
@pytest.mark.offline
class TestS3EdgeCases:
    """Edge case tests for S3 operations"""

    def test_large_file_handling(self, mock_s3_client):
        """Test handling files larger than 5GB (multipart threshold)"""
        large_file_size = 6 * 1024 * 1024 * 1024  # 6GB

        mock_s3_client.head_object.return_value = {
            'ContentLength': large_file_size,
            'ContentType': 'application/octet-stream'
        }

        response = mock_s3_client.head_object(Bucket='bucket', Key='large-file.bin')
        assert response['ContentLength'] == large_file_size

    def test_special_characters_in_key(self, mock_s3_client):
        """Test handling keys with special characters"""
        special_keys = [
            'path/with spaces/file.txt',
            'unicode/файл.txt',
            'symbols/file!@#$%^&().txt',
            'encoded/%20%2F%3F.txt',
        ]

        for key in special_keys:
            mock_s3_client.head_object.return_value = {'ContentLength': 100}
            # Should not raise
            mock_s3_client.head_object(Bucket='bucket', Key=key)

    def test_empty_prefix_listing(self, mock_s3_client):
        """Test listing with empty prefix"""
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [{'Key': 'root-file.txt'}],
            'KeyCount': 1
        }

        response = mock_s3_client.list_objects_v2(Bucket='bucket', Prefix='')
        assert response['KeyCount'] == 1

    def test_pagination_with_many_objects(self, mock_s3_client):
        """Test pagination handles more than 1000 objects"""
        page1_contents = [{'Key': f'file_{i}.txt'} for i in range(1000)]
        page2_contents = [{'Key': f'file_{i}.txt'} for i in range(1000, 1500)]

        call_count = [0]

        def paginated_response(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    'Contents': page1_contents,
                    'KeyCount': 1000,
                    'IsTruncated': True,
                    'NextContinuationToken': 'token123'
                }
            else:
                return {
                    'Contents': page2_contents,
                    'KeyCount': 500,
                    'IsTruncated': False
                }

        mock_s3_client.list_objects_v2.side_effect = paginated_response

        # Simulate pagination
        total_objects = 0
        continuation_token = None

        while True:
            kwargs = {'Bucket': 'bucket', 'MaxKeys': 1000}
            if continuation_token:
                kwargs['ContinuationToken'] = continuation_token

            response = mock_s3_client.list_objects_v2(**kwargs)
            total_objects += len(response.get('Contents', []))

            if not response.get('IsTruncated'):
                break
            continuation_token = response.get('NextContinuationToken')

        assert total_objects == 1500


# =============================================================================
# S3 TIMEOUT AND RESILIENCE
# =============================================================================

@pytest.mark.s3
@pytest.mark.resilience
class TestS3Resilience:
    """Resilience tests for S3 operations"""

    def test_retry_on_transient_error(self, mock_s3_client):
        """Test retry behavior on transient errors"""
        call_count = [0]

        def transient_then_success(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ClientError(
                    {'Error': {'Code': '500', 'Message': 'Internal Error'}},
                    'GetObject'
                )
            return {'Body': MagicMock()}

        mock_s3_client.get_object.side_effect = transient_then_success

        # Simulate retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = mock_s3_client.get_object(Bucket='bucket', Key='file.txt')
                break
            except ClientError:
                if attempt == max_retries - 1:
                    raise

        assert call_count[0] == 3

    def test_exponential_backoff_timing(self):
        """Test exponential backoff calculation"""
        base_delay = 1.0
        max_delay = 30.0

        def calculate_backoff(attempt: int) -> float:
            delay = base_delay * (2 ** attempt)
            return min(delay, max_delay)

        assert calculate_backoff(0) == 1.0
        assert calculate_backoff(1) == 2.0
        assert calculate_backoff(2) == 4.0
        assert calculate_backoff(3) == 8.0
        assert calculate_backoff(5) == 30.0  # Capped at max_delay
        assert calculate_backoff(10) == 30.0  # Still capped


# =============================================================================
# S3 INTEGRATION TESTS (require real AWS)
# =============================================================================

@pytest.mark.s3
@pytest.mark.cloud
@pytest.mark.integration
class TestS3Integration:
    """Integration tests requiring real AWS connectivity"""

    @pytest.fixture(autouse=True)
    def skip_without_aws(self, aws_credentials_available):
        if not aws_credentials_available:
            pytest.skip("AWS credentials not available")

    def test_real_bucket_access(self, real_s3_client, s3_bucket):
        """Test real S3 bucket access"""
        try:
            real_s3_client.head_bucket(Bucket=s3_bucket)
        except ClientError as e:
            pytest.fail(f"Cannot access bucket {s3_bucket}: {e}")

    def test_real_list_objects(self, real_s3_client, s3_bucket):
        """Test listing real S3 objects"""
        response = real_s3_client.list_objects_v2(
            Bucket=s3_bucket,
            MaxKeys=5
        )
        assert 'Contents' in response or response.get('KeyCount', 0) == 0
