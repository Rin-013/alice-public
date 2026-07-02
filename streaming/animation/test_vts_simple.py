#!/usr/bin/env python3
"""
Simple test: Can we move Alice at all?
Sends basic head movement to VTS to verify the connection works.
"""

import asyncio
import websockets
import json
import time
import math


async def test_vts_movement():
    """Connect to VTS and move the head back and forth."""

    # Connect
    uri = "ws://localhost:8001"
    print(f"Connecting to {uri}...")
    ws = await websockets.connect(uri)
    print("✓ Connected")

    # Authenticate
    print("Authenticating...")
    await ws.send(json.dumps({
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "auth-token",
        "messageType": "AuthenticationTokenRequest",
        "data": {"pluginName": "Alice Test", "pluginDeveloper": "Alice"}
    }))
    resp = json.loads(await ws.recv())
    token = resp.get("data", {}).get("authenticationToken")

    if not token:
        print("✗ Failed to get token")
        return

    await ws.send(json.dumps({
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "auth",
        "messageType": "AuthenticationRequest",
        "data": {
            "pluginName": "Alice Test",
            "pluginDeveloper": "Alice",
            "authenticationToken": token
        }
    }))
    resp = json.loads(await ws.recv())

    if not resp.get("data", {}).get("authenticated"):
        print("✗ Authentication failed")
        return

    print("✓ Authenticated")
    print("\nMoving Alice's head side to side...")
    print("Watch VTube Studio - her head should turn left and right")
    print("Press Ctrl+C to stop\n")

    # Move head side to side
    t = 0
    try:
        while True:
            # Simple sine wave for head angle X (yaw)
            angle_x = math.sin(t) * 15.0  # -15 to +15 degrees

            # Send parameter
            await ws.send(json.dumps({
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": "inject",
                "messageType": "InjectParameterDataRequest",
                "data": {
                    "parameterValues": [
                        {"id": "FaceAngleX", "value": angle_x, "weight": 1.0}
                    ]
                }
            }))

            # Receive response (required by API)
            await ws.recv()

            # Print status
            print(f"\rHead angle: {angle_x:6.2f}°", end="")

            t += 0.1
            await asyncio.sleep(0.033)  # ~30 FPS

    except KeyboardInterrupt:
        print("\n\nStopped")
    finally:
        await ws.close()


if __name__ == '__main__':
    asyncio.run(test_vts_movement())
