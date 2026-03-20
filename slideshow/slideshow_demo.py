from pathlib import Path
import json

from sanic import Sanic
from sanic.response import html
from websockets.exceptions import ConnectionClosed


SLIDESHOW_ROOT = Path(__file__).parent / "slideshow"
SLIDESHOW_HTML = SLIDESHOW_ROOT / "slideshow.html"

app = Sanic("slideshow_server")
app.static("/static", SLIDESHOW_ROOT)
app.ctx.websocket_clients = []


@app.get("/")
async def index(_request):
    return html(SLIDESHOW_HTML.read_text(encoding="utf-8"))


@app.websocket("/events")
async def events(_request, ws):
    app.ctx.websocket_clients.append(ws)
    print("WebSocket client connected.")

    try:
        while True:
            await ws.recv()
    except ConnectionClosed:
        print("WebSocket client disconnected.")
    finally:
        if ws in app.ctx.websocket_clients:
            app.ctx.websocket_clients.remove(ws)


@app.get("/command/<action>")
async def command(_request, action: str):
    print(f"Received command: {action}")
    await broadcast(json.dumps(action))
    return html("OK")


@app.listener("before_server_stop")
async def clean_sockets(app, _loop):
    for ws in list(app.ctx.websocket_clients):
        try:
            await ws.close()
        except Exception:
            pass


async def broadcast(message: str):
    disconnected = []

    for ws in app.ctx.websocket_clients:
        try:
            await ws.send(message)
        except ConnectionClosed:
            disconnected.append(ws)

    for ws in disconnected:
        if ws in app.ctx.websocket_clients:
            app.ctx.websocket_clients.remove(ws)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)