from typing import Optional
import click
from core.config import settings


@click.group()
def cli():
    """Ryoma CLI - AI Agent Runtime Tools

    Usage:
        Ryoma serve      Start API server
        Ryoma --help    Show help message
    """
    pass


@cli.command()
@click.option("--host", default=settings.API_HOST, help="Server listen address")
@click.option("--port", default=settings.API_PORT, help="Server listen port")
@click.option("--reload", is_flag=True, help="Enable auto-reload (dev mode)")
def serve(host: str, port: int, reload: Optional[bool] = False):
    """Start Ryoma API server"""
    import uvicorn

    uvicorn.run("ryoma.app:app", host=host, port=port, reload=reload)
