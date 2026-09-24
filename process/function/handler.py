"""HTTP handler: POST an image, get the stylized JPEG back.

Runs behind OpenFaaS of-watchdog in HTTP mode. Models are loaded once when
the process starts and reused for every request.

Request headers:
  X-Style-Name    "monet" (10 styles) or "varied" (32 styles). Default: varied.
  X-Style-Index   Style to apply. Default: 1.
  X-Which-Styles  Optional blend of styles, e.g. {3: 0.5, 7: 0.5}.
                  Overrides X-Style-Index.
"""

import ast
import io
import os

import numpy as np
from flask import Flask, Response, request
from PIL import Image, ImageOps

from function.model import Stylizer

MODEL_DIR = os.environ.get('MODEL_DIR', '/models')
# Inference cost grows with pixel count; big phone photos are downscaled.
MAX_IMAGE_SIZE = int(os.environ.get('MAX_IMAGE_SIZE', '1600'))

STYLIZERS = {
    name: Stylizer(os.path.join(
        MODEL_DIR, 'multistyle-pastiche-generator-%s.ckpt' % name))
    for name in ('monet', 'varied')
}

app = Flask(__name__)


class BadRequest(Exception):
    pass


def _parse_weights(headers):
    which_styles = headers.get('X-Which-Styles')
    if which_styles is None:
        try:
            return {int(headers.get('X-Style-Index', '1')): 1.0}
        except ValueError:
            raise BadRequest('X-Style-Index must be an integer')
    try:
        weights = ast.literal_eval(which_styles)
    except (ValueError, SyntaxError):
        raise BadRequest('X-Which-Styles must look like {3: 0.5, 7: 0.5}')
    if not isinstance(weights, dict) or not weights:
        raise BadRequest('X-Which-Styles must be a non-empty dict of '
                         'style index to weight; request one image per call')
    try:
        return {int(k): float(v) for k, v in weights.items()}
    except (TypeError, ValueError):
        raise BadRequest('X-Which-Styles keys must be ints, values numbers')


def _load_image(data):
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image).convert('RGB')
    except Exception:
        raise BadRequest('request body is not a readable image')
    image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    return np.asarray(image, dtype=np.float32) / 255.0


@app.route('/', methods=['POST'])
def stylize():
    try:
        style_name = request.headers.get('X-Style-Name', 'varied').lower()
        stylizer = STYLIZERS.get(style_name)
        if stylizer is None:
            raise BadRequest('X-Style-Name must be "monet" or "varied"')
        weights = _parse_weights(request.headers)
        image = _load_image(request.get_data())
        try:
            result = stylizer.stylize(image, weights)
        except ValueError as e:
            raise BadRequest(str(e))
    except BadRequest as e:
        return Response(str(e) + '\n', status=400, mimetype='text/plain')

    out = io.BytesIO()
    Image.fromarray(np.uint8(np.clip(result, 0, 1) * 255)).save(
        out, format='JPEG', quality=92)
    return Response(out.getvalue(), mimetype='image/jpeg')


@app.route('/', methods=['GET'])
def usage():
    return Response(__doc__, mimetype='text/plain')
