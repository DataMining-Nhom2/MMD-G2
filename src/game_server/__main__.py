"""Run game server."""
import uvicorn
import os

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    uvicorn.run(
        "game_server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
