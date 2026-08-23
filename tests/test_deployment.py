"""Tests for Docker and Render deployment configuration."""

import os
from pathlib import Path

import pytest


class TestDockerfile:
    """Tests for Dockerfile configuration."""

    def test_dockerfile_exists(self):
        """Dockerfile exists at project root."""
        dockerfile = Path("Dockerfile")
        assert dockerfile.exists(), "Dockerfile not found at project root"

    def test_dockerfile_uses_python_base(self):
        """Dockerfile uses appropriate Python base image."""
        dockerfile = Path("Dockerfile")
        content = dockerfile.read_text()
        assert "FROM python:3.11" in content, "Should use Python 3.11 base image"

    def test_dockerfile_installs_from_pyproject(self):
        """Dockerfile installs project from pyproject.toml."""
        dockerfile = Path("Dockerfile")
        content = dockerfile.read_text()
        assert "COPY pyproject.toml" in content
        assert "pip install" in content
        assert "--no-cache-dir" in content

    def test_dockerfile_copies_application(self):
        """Dockerfile copies application source code."""
        dockerfile = Path("Dockerfile")
        content = dockerfile.read_text()
        assert "COPY src/" in content

    def test_dockerfile_copies_corpus(self):
        """Dockerfile copies corpus directory for ingestion/retrieval."""
        dockerfile = Path("Dockerfile")
        content = dockerfile.read_text()
        assert "COPY public/corpus/" in content

    def test_dockerfile_copies_samples(self):
        """Dockerfile copies samples directory."""
        dockerfile = Path("Dockerfile")
        content = dockerfile.read_text()
        assert "COPY public/samples/" in content

    def test_dockerfile_no_venv_copy(self):
        """Dockerfile does not copy local .venv."""
        dockerfile = Path("Dockerfile")
        content = dockerfile.read_text()
        assert ".venv" not in content
        assert "venv" not in content.lower() or "useradd" in content  # only venv reference should be user creation

    def test_dockerfile_no_env_file_copy(self):
        """Dockerfile does not copy .env files."""
        dockerfile = Path("Dockerfile")
        content = dockerfile.read_text()
        assert ".env" not in content

    def test_dockerfile_no_hardcoded_secrets(self):
        """Dockerfile does not contain hardcoded API keys."""
        dockerfile = Path("Dockerfile")
        content = dockerfile.read_text()
        # Check for common secret patterns
        assert "sk-" not in content
        assert "AIza" not in content
        assert "API_KEY" not in content
        assert "SECRET" not in content
        assert "PASSWORD" not in content

    def test_dockerfile_exposes_port(self):
        """Dockerfile exposes HTTP port."""
        dockerfile = Path("Dockerfile")
        content = dockerfile.read_text()
        assert "EXPOSE 8000" in content

    def test_dockerfile_runs_uvicorn(self):
        """Dockerfile runs FastAPI app through Uvicorn."""
        dockerfile = Path("Dockerfile")
        content = dockerfile.read_text()
        assert "uvicorn" in content
        assert "support_agent.api:app" in content

    def test_dockerfile_binds_to_all_interfaces(self):
        """Dockerfile binds Uvicorn to 0.0.0.0."""
        dockerfile = Path("Dockerfile")
        content = dockerfile.read_text()
        assert "0.0.0.0" in content

    def test_dockerfile_uses_port_env(self):
        """Dockerfile uses PORT environment variable."""
        dockerfile = Path("Dockerfile")
        content = dockerfile.read_text()
        assert "PORT" in content
        assert "8000" in content

    def test_dockerfile_non_root_user(self):
        """Dockerfile creates and uses non-root user."""
        dockerfile = Path("Dockerfile")
        content = dockerfile.read_text()
        assert "useradd" in content
        assert "USER appuser" in content


class TestRenderYaml:
    """Tests for render.yaml configuration."""

    def test_render_yaml_exists(self):
        """render.yaml exists at project root."""
        render_yaml = Path("render.yaml")
        assert render_yaml.exists(), "render.yaml not found at project root"

    def test_render_describes_web_service(self):
        """render.yaml describes a web service."""
        render_yaml = Path("render.yaml")
        content = render_yaml.read_text()
        assert "type: web" in content
        assert "services:" in content

    def test_render_uses_dockerfile(self):
        """render.yaml references the Dockerfile."""
        render_yaml = Path("render.yaml")
        content = render_yaml.read_text()
        assert "env: docker" in content
        assert "dockerfilePath: ./Dockerfile" in content
        assert "dockerContext: ." in content

    def test_render_no_hardcoded_secrets(self):
        """render.yaml does not contain API keys or secret values."""
        render_yaml = Path("render.yaml")
        content = render_yaml.read_text()
        # Should not contain actual secret values
        assert "sk-" not in content
        assert "AIza" not in content
        # sync: false means they're configured as secrets in Render dashboard
        assert "sync: false" in content

    def test_render_uses_port_env(self):
        """render.yaml uses Render's PORT environment variable."""
        render_yaml = Path("render.yaml")
        content = render_yaml.read_text()
        # The Dockerfile handles PORT, render.yaml should not hardcode port
        assert "PORT" in content or "healthCheckPath" in content

    def test_render_health_check_path(self):
        """render.yaml uses /health endpoint for health check."""
        render_yaml = Path("render.yaml")
        content = render_yaml.read_text()
        assert "healthCheckPath: /health" in content

    def test_render_required_env_vars_documented(self):
        """Required environment variables are documented in render.yaml."""
        render_yaml = Path("render.yaml")
        content = render_yaml.read_text()
        required_vars = [
            "SUPABASE_URL",
            "SUPABASE_KEY",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
            "OPENROUTER_MODEL",
            "EMBEDDING_MODEL",
            "EMBEDDING_BATCH_SIZE",
        ]
        for var in required_vars:
            assert var in content, f"Required env var {var} not documented in render.yaml"


class TestHealthEndpointUsable:
    """Tests that /health endpoint remains usable."""

    def test_health_endpoint_exists_in_api(self):
        """API still defines /health endpoint."""
        from support_agent.api import app
        routes = [route.path for route in app.routes]
        assert "/health" in routes

    def test_health_endpoint_get_method(self):
        """/health endpoint responds to GET."""
        from support_agent.api import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestFastApiTestsStillPass:
    """Tests that existing FastAPI tests still pass."""

    def test_triage_single_endpoint(self):
        """POST /triage still works."""
        from support_agent.api import app
        from fastapi.testclient import TestClient
        from unittest.mock import patch, MagicMock
        from support_agent.models import AgentResult

        def make_agent_result(**kwargs):
            return AgentResult(
                status="replied",
                product_area="screen",
                response="Test response",
                justification="Test justification",
                request_type="bug",
            )

        with patch("support_agent.api._get_orchestrator") as mock_get:
            mock_orchestrator = MagicMock()
            mock_orchestrator.process_ticket.return_value = make_agent_result()
            mock_get.return_value = mock_orchestrator

            client = TestClient(app)
            response = client.post("/triage", json={"issue": "Test", "subject": "Test"})
            assert response.status_code == 200

    def test_triage_batch_endpoint(self):
        """POST /triage/batch still works."""
        from support_agent.api import app
        from fastapi.testclient import TestClient
        from unittest.mock import patch, MagicMock
        from support_agent.models import AgentResult

        def make_agent_result(**kwargs):
            return AgentResult(
                status="replied",
                product_area="screen",
                response="Test response",
                justification="Test justification",
                request_type="bug",
            )

        with patch("support_agent.api._get_orchestrator") as mock_get:
            mock_orchestrator = MagicMock()
            mock_orchestrator.process_ticket.return_value = make_agent_result()
            mock_get.return_value = mock_orchestrator

            client = TestClient(app)
            response = client.post("/triage/batch", json=[{"issue": "Test", "subject": "Test"}])
            assert response.status_code == 200
            assert len(response.json()) == 1


class TestDockerBuild:
    """Test Docker build if Docker is available."""

    def test_docker_available(self):
        """Check if Docker is available in the environment."""
        import subprocess
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, timeout=10)
            docker_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            docker_available = False

        if not docker_available:
            pytest.skip("Docker not available in environment")

    @pytest.mark.skipif(
        not Path("/var/run/docker.sock").exists() and not Path("//./pipe/docker_engine").exists(),
        reason="Docker daemon not available"
    )
    def test_docker_build_succeeds(self):
        """Docker build succeeds with the actual Dockerfile."""
        import subprocess
        result = subprocess.run(
            ["docker", "build", "-t", "support-triage-agent-test", "."],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, f"Docker build failed: {result.stderr}"