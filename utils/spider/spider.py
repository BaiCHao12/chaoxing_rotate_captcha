# 爬取学习通旋转验证码图片
import json
import os
import random
import string
import time
from pathlib import Path

import requests
from encrypt import generate_captcha_key
from tqdm import trange

SHADE_IMAGE_PATH = Path.cwd() / "imgs_3"
CUTOUT_IMAGE_PATH = os.path.join(SHADE_IMAGE_PATH, "border")
Path(CUTOUT_IMAGE_PATH).mkdir(parents=True, exist_ok=True)


class Spider:
    def __init__(self):
        self.session = requests.session()
        # 设置会话ua
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0"
            }
        )
        self.init_cookie()

    def init_cookie(self):
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "Cache-Control": "max-age=0",
            "Connection": "keep-alive",
            "Referer": "https://cn.bing.com/",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
        }

        response = self.session.get("https://v8.chaoxing.com/", headers=headers)
        return response

    def getCaptcha(self):
        headers = {
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "Connection": "keep-alive",
            "Referer": "https://v8.chaoxing.com/",
            "Sec-Fetch-Dest": "script",
            "Sec-Fetch-Mode": "no-cors",
            "Sec-Fetch-Site": "same-site",
        }
        captchaId = "qDG21VMg9qS5Rcok4cfpnHGnpf5LhcAv"
        captchaType = "rotate"
        capture_key, token = generate_captcha_key(
            timestamp=int(time.time() * 1000),
            captchaId=captchaId,
            type=captchaType,
        )
        callback = f"jQuery3610{''.join(random.choices(string.digits, k=16))}_{str(int(time.time() * 1000))}"
        params = {
            "callback": callback,
            "captchaId": captchaId,
            "type": captchaType,
            "version": "1.1.16",
            "captchaKey": capture_key,
            "token": token,
            "referer": "https://v8.chaoxing.com/",
            "_": str(int(time.time() * 1000)),
        }

        response = self.session.get(
            "https://captcha.chaoxing.com/captcha/get/verification/image",
            params=params,
            headers=headers,
        )
        json_str = response.text.replace(callback + "(", "")
        # 去除字符串最后一个字符
        json_str = json_str[:-1]

        return json.loads(json_str)

    def download_image(self, url, path):
        """根据url下载图片"""
        response = self.session.get(url)

        with open(
            os.path.join(path, url.split("/")[-1:][0].replace("jpg", "png")), "wb"
        ) as f:
            f.write(response.content)


if __name__ == "__main__":
    spider = Spider()
    for i in trange(100):
        captcha = spider.getCaptcha()["imageVerificationVo"]
        spider.download_image(captcha["shadeImage"], SHADE_IMAGE_PATH)
        spider.download_image(captcha["cutoutImage"], CUTOUT_IMAGE_PATH)
        # 随机等待20-30秒
        time.sleep(random.randint(20, 30))
