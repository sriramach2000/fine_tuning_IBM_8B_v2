"""
Concurrency Test Suite

Tests for parallel request handling including:
- Thread pool execution
- Race condition detection
- Concurrent S3 access
- Batch processing integrity
"""

import json
import time
import threading
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock


# =============================================================================
# THREAD POOL TESTS
# =============================================================================

@pytest.mark.concurrency
@pytest.mark.offline
class TestThreadPoolExecution:
    """Tests for ThreadPoolExecutor behavior"""

    def test_parallel_request_completion(self):
        """Test all parallel requests complete"""
        num_requests = 10
        results = []
        lock = threading.Lock()

        def mock_request(request_id):
            time.sleep(0.01)  # Simulate network latency
            with lock:
                results.append(request_id)
            return request_id

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(mock_request, i) for i in range(num_requests)]
            completed = [f.result() for f in as_completed(futures)]

        assert len(completed) == num_requests
        assert len(results) == num_requests

    def test_results_order_preserved_after_sort(self):
        """Test results can be sorted to preserve order"""
        results = []

        def process(item_id):
            time.sleep(0.01 * (10 - item_id))  # Reverse order completion
            return {'id': item_id, 'data': f'result_{item_id}'}

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process, i) for i in range(10)]
            for future in as_completed(futures):
                results.append(future.result())

        # Sort by ID to restore order
        sorted_results = sorted(results, key=lambda x: x['id'])

        assert [r['id'] for r in sorted_results] == list(range(10))

    def test_exception_handling_in_threads(self):
        """Test exceptions in threads are properly propagated"""
        def failing_task(should_fail):
            if should_fail:
                raise ValueError("Task failed")
            return "success"

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(failing_task, False),
                executor.submit(failing_task, True),
                executor.submit(failing_task, False),
            ]

            successes = 0
            failures = 0

            for future in as_completed(futures):
                try:
                    future.result()
                    successes += 1
                except ValueError:
                    failures += 1

        assert successes == 2
        assert failures == 1


# =============================================================================
# RACE CONDITION TESTS
# =============================================================================

@pytest.mark.concurrency
@pytest.mark.offline
class TestRaceConditions:
    """Tests for race condition detection"""

    def test_counter_with_lock(self):
        """Test counter increment with proper locking"""
        counter = [0]
        lock = threading.Lock()

        def increment():
            for _ in range(100):
                with lock:
                    counter[0] += 1

        threads = [threading.Thread(target=increment) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert counter[0] == 1000

    def test_results_list_thread_safety(self):
        """Test thread-safe list appending"""
        results = []
        lock = threading.Lock()

        def append_result(value):
            time.sleep(0.001)
            with lock:
                results.append(value)

        threads = [threading.Thread(target=append_result, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 100
        assert set(results) == set(range(100))


# =============================================================================
# BATCH PROCESSING TESTS
# =============================================================================

@pytest.mark.concurrency
@pytest.mark.offline
class TestBatchProcessing:
    """Tests for batch processing integrity"""

    def test_batch_results_match_inputs(self, sample_prompts):
        """Test all batch inputs produce outputs"""
        def process_prompt(prompt):
            return {
                'id': prompt['id'],
                'response': f"Response for {prompt['id']}",
                'success': True
            }

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(process_prompt, p): p['id']
                for p in sample_prompts
            }

            results = {}
            for future in as_completed(futures):
                result = future.result()
                results[result['id']] = result

        # All inputs should have outputs
        for prompt in sample_prompts:
            assert prompt['id'] in results

    def test_partial_failure_handling(self, sample_prompts):
        """Test batch continues despite partial failures"""
        call_count = [0]

        def process_with_failure(prompt):
            call_count[0] += 1
            if call_count[0] == 2:  # Fail second request
                return {'id': prompt['id'], 'success': False, 'error': 'Simulated failure'}
            return {'id': prompt['id'], 'success': True, 'response': 'OK'}

        with ThreadPoolExecutor(max_workers=1) as executor:  # Sequential for predictability
            futures = [executor.submit(process_with_failure, p) for p in sample_prompts]
            results = [f.result() for f in futures]

        successes = sum(1 for r in results if r['success'])
        failures = sum(1 for r in results if not r['success'])

        assert successes == 2
        assert failures == 1
        assert len(results) == 3  # All processed


# =============================================================================
# CONCURRENT AWS ACCESS
# =============================================================================

@pytest.mark.concurrency
@pytest.mark.offline
class TestConcurrentAWSAccess:
    """Tests for concurrent AWS service access"""

    def test_concurrent_s3_reads(self, mock_s3_client):
        """Test concurrent S3 read operations"""
        call_count = [0]
        lock = threading.Lock()

        def mock_get_object(*args, **kwargs):
            with lock:
                call_count[0] += 1
            time.sleep(0.01)  # Simulate latency
            body = MagicMock()
            body.read.return_value = b'file content'
            return {'Body': body}

        mock_s3_client.get_object.side_effect = mock_get_object

        def read_file(key):
            return mock_s3_client.get_object(Bucket='bucket', Key=key)

        keys = [f'file_{i}.txt' for i in range(10)]

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(read_file, key) for key in keys]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 10
        assert call_count[0] == 10

    def test_concurrent_bedrock_invocations(self, mock_bedrock_client):
        """Test concurrent Bedrock model invocations"""
        call_count = [0]
        lock = threading.Lock()

        def mock_invoke(*args, **kwargs):
            with lock:
                call_count[0] += 1
            time.sleep(0.01)
            body = MagicMock()
            body.read.return_value = json.dumps({
                'content': [{'text': 'Response'}],
                'usage': {'input_tokens': 10, 'output_tokens': 50}
            }).encode()
            return {'body': body}

        mock_bedrock_client.invoke_model.side_effect = mock_invoke

        def invoke_model(prompt):
            return mock_bedrock_client.invoke_model(
                modelId='test-model',
                body=json.dumps({'messages': [{'role': 'user', 'content': prompt}]})
            )

        prompts = [f'Prompt {i}' for i in range(5)]

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(invoke_model, p) for p in prompts]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 5
        assert call_count[0] == 5
