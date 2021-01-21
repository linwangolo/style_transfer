import json
import numpy as np
import base64
import argparse
import asyncio
from aiohttp import web
import aiohttp_cors
import cv2

import style_transfer


async def transfer(request):
    async with sem:
        data = await request.json()  #.read()
        request_ip = request.remote
        print(request_ip)

        data = np.fromstring(base64.decodestring(data['image_file_b64'].encode()), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        try:
            await model_predict(img=img, result_dir=f'/home/at/lin/style_result_test_api', file_name=f'{request_ip}.jpg') #data['result_dir'])
            test_img = cv2.imread('/home/at/lin/style_result_test_api/result-toon.jpg')
            img_str = cv2.imencode('.jpg', test_img)[1].tostring()
        except:
            img_str = cv2.imencode('.jpg', img)[1].tostring()

        resp = web.Response(body=json.dumps({'resullt':'OK'}))
        resp.content_type = 'application/json'

        # resp = web.Response(body=img_str)
        # resp.content_type = 'image/*'
        # print('returning transferred image ...')
        return resp

async def model_predict(img, result_dir, file_name):
    trans.transfer(img=img, result_dir=result_dir, file_name=file_name) 

async def reload_to_get_image(request):
    async with sem:
        request_ip = request.remote
        print(request_ip)

        img = cv2.imread(f'/home/at/lin/style_result_test_api/{request_ip}.jpg')
        img_str = cv2.imencode('.jpg', img)[1].tostring()

        resp = web.Response(body=img_str)
        resp.content_type = 'image/*'
        print('returning transferred image ...')
        return resp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=8000, type=int)
    parser.add_argument('--thread_num', default=3, type=int)
    arg = parser.parse_args()

    trans = style_transfer.StyleTransfer()

    sem = asyncio.Semaphore(arg.thread_num)
    
    app = web.Application(client_max_size=1024**2*4)
    # app.add_routes([web.post('/transfer', transfer)])

    cors = aiohttp_cors.setup(app, defaults={
       "*": aiohttp_cors.ResourceOptions(
           allow_credentials=True,
           expose_headers="*",
           allow_headers="*",
       )
   })
    resource = cors.add(app.router.add_resource("/transfer"))
    cors.add(resource.add_route("POST", transfer))

    resource = cors.add(app.router.add_resource("/reload"))
    cors.add(resource.add_route("POST", reload_to_get_image))

    web.run_app(app, port=arg.port)
