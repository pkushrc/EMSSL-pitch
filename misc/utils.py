# coding = utf-8
from __future__ import division
from __future__ import print_function

import os
import re
import errno

def mkdir_p(path):
    try: 
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_filename(fp):
    regex = "([^/]+)$"
    matchObj = re.search(regex, fp)
    if matchObj:
        filename = matchObj.group()
    else:
        raise ValueError("No match filename in {}".format(fp))
    return filename

def get_trm_filename(audio_fp):
    '''
    Get trm filenames from audio file path 
    '''
    audio_filename = get_filename(audio_fp)
    return audio_filename.split('.')[0].strip('_') + '_trm_params.txt'