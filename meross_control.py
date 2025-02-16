import asyncio
from meross_iot.http_api import MerossHttpClient
from meross_iot.manager import MerossManager

async def meross_toggle():
    email = "joseph.shunk@gmail.com"
    password = "MerossSphere7$"
    http_api_client = await MerossHttpClient.async_from_user_password(api_base_url='https://iotx-us.meross.com',
                                                                      email=email, 
                                                                      password=password)

    # Setup and start the device manager
    manager = MerossManager(http_client=http_api_client)
    await manager.async_init()

    # Retrieve all the MSS310 devices that are registered on this account
    await manager.async_device_discovery()
    devices = manager.find_devices()

    if len(devices) < 1:
        print("No devices found...")
    else:
        # Turn it on channel 0
        # Note that channel argument is optional for MSS310 as they only have one channel
        dev = devices[0]

        # The first time we play with a device, we must update its status
        await dev.async_update()

        # We can now start playing with that
        print(f"Turning on {dev.name}...")
        await dev.async_turn_on(channel=0)
        print("Waiting a bit before turing it off")
        await asyncio.sleep(5)
        print(f"Turing off {dev.name}")
        await dev.async_turn_off(channel=0)

    # Close the manager and logout from http_api
    manager.close()
    await http_api_client.async_logout()