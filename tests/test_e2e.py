from fastapi.testclient import TestClient

from emma_perception.commands.run_server import app


def test_api_runs_without_crashing():
    """Verify that the API runs without crashing."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "success"
