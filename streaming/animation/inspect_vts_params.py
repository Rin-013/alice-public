#!/usr/bin/env python3
"""
Connect to VTube Studio and inspect available parameters.

This shows us the ACTUAL parameter schema used by the current model,
so we can ensure our data uses the correct param names.
"""

import asyncio
import websockets
import json
import time


class VTSInspector:
    """Simple VTS API client for parameter inspection."""

    def __init__(self, host="127.0.0.1", port=8001):
        self.host = host
        self.port = port
        self.websocket = None
        self.auth_token = None

    async def connect(self):
        """Connect to VTube Studio."""
        try:
            self.websocket = await websockets.connect(f"ws://{self.host}:{self.port}")
            print(f"✓ Connected to VTube Studio at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect: {e}")
            print("\nMake sure:")
            print("  1. VTube Studio is running")
            print("  2. API is enabled in Settings > General Settings")
            print("  3. Port is 8001 (default)")
            return False

    async def send_request(self, message_type: str, data: dict = None):
        """Send a request to VTS API."""
        request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": f"inspect_{int(time.time() * 1000)}",
            "messageType": message_type
        }
        if data:
            request["data"] = data

        await self.websocket.send(json.dumps(request))
        response = await self.websocket.recv()
        return json.loads(response)

    async def authenticate(self):
        """Authenticate with VTS (required for most API calls)."""
        # Request authentication
        auth_request = await self.send_request("AuthenticationTokenRequest", {
            "pluginName": "Alice Motion Inspector",
            "pluginDeveloper": "Alice AI"
        })

        if "data" in auth_request and "authenticationToken" in auth_request["data"]:
            self.auth_token = auth_request["data"]["authenticationToken"]
            print(f"✓ Got authentication token")

            # Actually authenticate
            auth_response = await self.send_request("AuthenticationRequest", {
                "pluginName": "Alice Motion Inspector",
                "pluginDeveloper": "Alice AI",
                "authenticationToken": self.auth_token
            })

            if auth_response.get("data", {}).get("authenticated"):
                print(f"✓ Authenticated successfully")
                return True

        print(f"✗ Authentication failed")
        return False

    async def get_current_model(self):
        """Get the currently loaded model info."""
        response = await self.send_request("CurrentModelRequest")
        if "data" in response:
            model = response["data"]
            print(f"\n=== Current Model ===")
            print(f"Name: {model.get('modelName', 'Unknown')}")
            print(f"ID: {model.get('modelID', 'Unknown')}")
            print(f"Loaded: {model.get('modelLoaded', False)}")
            return model
        return None

    async def get_input_parameters(self):
        """Get all available input parameters for the current model."""
        response = await self.send_request("InputParameterListRequest")

        if "data" in response and "customParameters" in response["data"]:
            params = response["data"]["customParameters"]
            default_params = response["data"].get("defaultParameters", [])

            print(f"\n=== Input Parameters ===")
            print(f"Default parameters: {len(default_params)}")
            print(f"Custom parameters: {len(params)}")
            print(f"Total: {len(default_params) + len(params)}")

            # Show default params (these are standard across models)
            print(f"\n--- Default Parameters ---")
            for param in default_params[:20]:  # Show first 20
                name = param.get('name', 'Unknown')
                value = param.get('value', 0.0)
                min_val = param.get('min', 0.0)
                max_val = param.get('max', 1.0)
                print(f"  {name:<20} = {value:7.3f}  (range: {min_val:.1f} to {max_val:.1f})")

            if len(default_params) > 20:
                print(f"  ... and {len(default_params) - 20} more")

            # Show custom params (model-specific)
            print(f"\n--- Custom Parameters (first 30) ---")
            for param in params[:30]:
                name = param.get('name', 'Unknown')
                value = param.get('value', 0.0)
                min_val = param.get('min', 0.0)
                max_val = param.get('max', 1.0)
                print(f"  {name:<30} = {value:7.3f}  (range: {min_val:.1f} to {max_val:.1f})")

            if len(params) > 30:
                print(f"  ... and {len(params) - 30} more")

            # Save full list
            all_params = default_params + params
            with open('streaming/animation/vts_parameters.json', 'w') as f:
                json.dump({
                    'default_parameters': default_params,
                    'custom_parameters': params,
                    'total_count': len(all_params)
                }, f, indent=2)

            print(f"\n✓ Full parameter list saved to streaming/animation/vts_parameters.json")

            return all_params

        return []

    async def disconnect(self):
        """Close connection."""
        if self.websocket:
            await self.websocket.close()
            print(f"\n✓ Disconnected")


async def main():
    inspector = VTSInspector()

    if not await inspector.connect():
        return 1

    if not await inspector.authenticate():
        print("\n⚠ Make sure you click 'Allow' in VTube Studio!")
        return 1

    # Get current model
    model = await inspector.get_current_model()

    if not model or not model.get('modelLoaded'):
        print("\n✗ No model loaded in VTube Studio!")
        return 1

    # Get all parameters
    params = await inspector.get_input_parameters()

    if not params:
        print("\n✗ Failed to get parameters")
        return 1

    print(f"\n=== Analysis ===")
    print(f"This model has {len(params)} total input parameters.")
    print(f"\nKey parameters for animation:")

    # Look for common animation params
    common_names = [
        'FaceAngleX', 'FaceAngleY', 'FaceAngleZ',
        'EyeOpenLeft', 'EyeOpenRight',
        'EyeLeftX', 'EyeRightX', 'EyeLeftY', 'EyeRightY',
        'MouthOpen', 'MouthSmile', 'MouthForm',
        'ParamAngleX', 'ParamAngleY', 'ParamAngleZ',
        'ParamEyeLOpen', 'ParamEyeROpen',
        'ParamEyeBallX', 'ParamEyeBallY',
        'ParamMouthOpenY', 'ParamMouthForm'
    ]

    found_params = []
    for param in params:
        name = param.get('name', '')
        if any(common in name for common in common_names):
            found_params.append(name)

    if found_params:
        print("\nFound animation parameters:")
        for name in sorted(found_params):
            print(f"  - {name}")
    else:
        print("\n⚠ No standard animation parameters found")
        print("  Model might use custom naming scheme")

    await inspector.disconnect()

    print("\n💡 Next steps:")
    print("  1. Check vts_parameters.json for full parameter list")
    print("  2. Update clip extraction to use these exact param names")
    print("  3. Ensure avatar CSV data matches VTS param names")

    return 0


if __name__ == '__main__':
    exit(asyncio.run(main()))
