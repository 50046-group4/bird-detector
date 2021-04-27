# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import asyncio

import aiohttp
import aioredis
import time

async def esp32_cam_fetcher(*, cam_host:str, cam_id:str, redis):
    while True:
        async with aiohttp.ClientSession() as session:
            async with session.get(cam_host) as response:
                reader = aiohttp.MultipartReader.from_response(response)
                while True:
                    part = await reader.next()
                    if part is None:
                        break
                    image = await part.read()
                    timestamp = str(round(time.time()))
                    await redis.set(f"sentry.{cam_id}.raw_images.{timestamp}", str(image), expire=30)
                    await redis.publish(f"sentry.{cam_id}.updates",timestamp)


async def main():
    redis = await aioredis.create_redis_pool("redis://redis")
    cam_fetchers = [
        esp32_cam_fetcher(cam_host="http://sentry_a_cam", cam_id="A", redis=redis),
        esp32_cam_fetcher(cam_host="http://sentry_b_cam", cam_id="B", redis=redis)
    ]
    await asyncio.gather(
        *cam_fetchers
    )

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    asyncio.run(main())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

