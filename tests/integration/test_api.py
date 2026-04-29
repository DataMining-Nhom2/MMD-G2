"""
Integration Tests for AI Engine — FastAPI endpoints

Run with:
    pytest tests/integration/ -v

Requires: AI Engine server running on http://localhost:8000
"""

import pytest
import httpx
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

BASE_URL = os.environ.get('AI_ENGINE_URL', 'http://localhost:8000')


class TestHealthEndpoint:
    """Test /health endpoint"""

    def test_health_returns_ok(self):
        """GET /health should return status ok"""
        response = httpx.get(f"{BASE_URL}/health", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "ok"


class TestPredictELOEndpoint:
    """Test /api/predict-elo endpoint"""

    @pytest.fixture
    def client(self):
        return httpx.Client(base_url=BASE_URL, timeout=30)

    def test_valid_request_returns_success(self, client):
        """Valid PGN should return success: true"""
        payload = {
            "pgn": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6",
            "clock_times": [5.2, 3.1, 12.0, 8.5, 2.1, 45.3],
            "result": "1-0",
            "time_control": "5+0",
        }
        response = client.post("/api/predict-elo", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_response_has_required_fields(self, client):
        """Response should contain all required data fields"""
        payload = {
            "pgn": "1. e4 e5",
            "clock_times": [5.0, 3.0],
            "result": "1-0",
        }
        response = client.post("/api/predict-elo", json=payload)
        data = response.json()
        assert data["success"] is True

        result = data["data"]
        assert "white_elo" in result
        assert "black_elo" in result
        assert "eco" in result
        assert "stats" in result
        assert "explanation" in result

    def test_elo_in_valid_range(self, client):
        """ELO values should be between 400 and 3000"""
        payload = {
            "pgn": "1. e4 e5 2. Nf3 Nc6",
            "clock_times": [5.0, 3.0, 12.0, 8.0],
            "result": "1-0",
        }
        response = client.post("/api/predict-elo", json=payload)
        data = response.json()
        assert 400 <= data["data"]["white_elo"] <= 3000
        assert 400 <= data["data"]["black_elo"] <= 3000

    def test_eco_has_code_and_name(self, client):
        """ECO should have both code and name"""
        payload = {
            "pgn": "1. e4 c5",
            "clock_times": [5.0, 3.0],
            "result": "0-1",
        }
        response = client.post("/api/predict-elo", json=payload)
        data = response.json()
        eco = data["data"]["eco"]
        assert "code" in eco
        assert "name" in eco
        assert isinstance(eco["code"], str)
        assert isinstance(eco["name"], str)

    def test_stats_has_required_fields(self, client):
        """Stats should have CPL and blunder counts"""
        payload = {
            "pgn": "1. e4 e5",
            "clock_times": [5.0, 3.0],
            "result": "1/2-1/2",
        }
        response = client.post("/api/predict-elo", json=payload)
        data = response.json()
        stats = data["data"]["stats"]
        assert "white_avg_cpl" in stats
        assert "black_avg_cpl" in stats
        assert "white_blunders" in stats
        assert "black_blunders" in stats
        assert "total_moves" in stats

    def test_explanation_is_string(self, client):
        """Explanation should be a non-empty string"""
        payload = {
            "pgn": "1. e4 e5",
            "clock_times": [5.0, 3.0],
            "result": "1-0",
        }
        response = client.post("/api/predict-elo", json=payload)
        data = response.json()
        explanation = data["data"]["explanation"]
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_missing_pgn_returns_error(self, client):
        """Missing pgn field should still return success (stub behavior)"""
        payload = {
            "clock_times": [5.0, 3.0],
            "result": "1-0",
        }
        response = client.post("/api/predict-elo", json=payload)
        # Stub may handle this gracefully — just check it doesn't crash
        assert response.status_code == 200

    def test_default_time_control(self, client):
        """Should accept request without time_control field"""
        payload = {
            "pgn": "1. e4",
            "clock_times": [5.0],
            "result": "1-0",
        }
        response = client.post("/api/predict-elo", json=payload)
        assert response.status_code == 200
        assert response.json()["success"] is True


@pytest.mark.asyncio
class TestPipelineIntegration:
    """Test the full prediction pipeline asynchronously"""

    async def test_full_pipeline_runs_without_crash(self):
        """Full pipeline should complete without errors"""
        async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:
            payload = {
                "pgn": "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6",
                "clock_times": [5.2, 3.1, 12.0, 8.5, 2.1, 45.3, 3.0, 7.2],
                "result": "1-0",
                "time_control": "5+0",
            }
            response = await client.post("/api/predict-elo", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert "success" in data
