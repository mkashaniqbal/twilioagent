import os
import json
import base64
import asyncio
import websockets
from openai import OpenAI
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from twilio.twiml.voice_response import VoiceResponse, Connect
import uvicorn
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

# Variables de configuration d'enregistrement
LOGS_DIRECTORY = "conversation_logs"
os.makedirs(LOGS_DIRECTORY, exist_ok=True)

SYSTEM_MESSAGE = (
    "Vous êtes un agent immobilier AI joyeux et serviable,\n"
    "spécialisé dans la location d'appartements cosy et abordables..."
)

VOICE = 'alloy'
LOG_EVENT_TYPES = [
    'response.content.done', 'rate_limits.updated',
    'response.done',
    'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped',
    'input_audio_buffer.speech_started', 'session.created'
]

app = FastAPI()

@app.api_route("/", methods=["GET", "POST"])
async def index_page():
    return "<h1>ça fonctionne</h1>"

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    response = VoiceResponse()
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f"wss://{host}/media-stream")
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="text/xml")

@app.post("/sync-call-history")
async def sync_call_history(request: Request):
    """Handle call history synchronization."""
    data = await request.json()
    print("Received sync-call-history:", data)
    return {"status": "received"}

async def send_session_update(ws):
    start_message = {
        "type": "start",
        "config": {
            "transcription_config": {
                "encoding": "audio/x-mulaw",
                "sample_rate_hz": 8000,
                "language_code": "fr-FR"
            },
            "response_format": "text"
        }
    }
    await ws.send(json.dumps(start_message))

    session_message = {
        "type": "session.update",
        "input_audio_config": {
            "encoding": "audio/x-mulaw",
            "sample_rate": 8000
        },
        "output_audio_config": {
            "encoding": "audio/x-mulaw",
            "sample_rate": 8000
        }
    }
    await ws.send(json.dumps(session_message))

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI."""
    print("Client connected")
    await websocket.accept()

    async with websockets.connect(
        'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17',
        additional_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
    ) as openai_ws:

        await send_session_update(openai_ws)
        stream_sid = None
        conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
        audio_log = open(os.path.join(LOGS_DIRECTORY, f"{conversation_id}_audio.txt"), "a")

        try:
            async def send_to_twilio():
                """Receive events from OpenAI, send audio back to Twilio, and log responses."""
                nonlocal stream_sid
                try:
                    async for openai_message in openai_ws:
                        response = json.loads(openai_message)
                        if response['type'] in LOG_EVENT_TYPES:
                            print(f"Received event: {response['type']}")
                        if response['type'] == 'session.updated':
                            print("Session updated successfully:", response)

                        if response['type'] == 'response.audio.delta' and response.get('delta'):
                            try:
                                audio_payload = base64.b64encode(
                                    base64.b64decode(response['delta'])
                                ).decode('utf-8')

                                audio_delta = {
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {
                                        "payload": audio_payload
                                    }
                                }
                                await websocket.send_json(audio_delta)
                                audio_log.write(f"Sent audio: {audio_payload}\n")

                            except Exception as e:
                                print(f"Error processing audio data: {e}")

                        if response['type'] == 'response.text' and response.get('text'):
                            print("AI Response:", response['text'])

                except Exception as e:
                    print(f"Error in send_to_twilio: {e}")

            async def receive_from_twilio():
                async for message in websocket.iter_text():
                    try:
                        data = json.loads(message)

                        if data['event'] == 'media':
                            audio_append = {
                                "type": "input_audio_buffer.append",
                                "audio": data['media']['payload']
                            }
                            msg = json.dumps(audio_append)
                            print(">> Sending:", msg)
                            await openai_ws.send(msg)
                            response = await openai_ws.recv()
                            print("<< Received:", response)
                            audio_log.write(f"Received audio: {data['media']['payload']}\n")

                        if data['event'] == 'start':
                            nonlocal stream_sid
                            stream_sid = data['start']['streamSid']
                            print(f"Incoming stream has started\n{stream_sid}")

                    except Exception as e:
                        print(f"Error processing message: {e}")
                        continue

            # Start bidirectional audio bridge
            await asyncio.gather(receive_from_twilio(), send_to_twilio())

        except WebSocketDisconnect:
            print("Client disconnected.")
            if openai_ws.open:
                await openai_ws.close()
        finally:
            audio_log.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 3000)))
