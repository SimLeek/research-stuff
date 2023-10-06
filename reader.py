"""
Usage:
  reader.py [--tempo=<tempo>] [--voice_model=<model>]

Options:
  -h --help                    Show this help message and exit.
  --tempo=<tempo>              Voice reading speed. [default: 1.4]
  --voice_model=<voice_model>  TTS model to use.
"""
import sys

import torch
from TTS.api import TTS
import tempfile
import numpy as np
import scipy
import subprocess
import os
from docopt import docopt
import string

dir_path = os.path.dirname(os.path.realpath(__file__))
# Get device. Use last GPU rather than the one probably used for the screen and every single other app
device = torch.cuda.device_count() - 1 if torch.cuda.is_available() else "cpu"

test_text = """
This is a test! This is also a test!!
This... is a test... ON ANOTHER LINE!
"""


def main(
    full_text=test_text,
    voice_model="tts_models/en/jenny/jenny",
    tempo=1.5,
):
    print(TTS().list_models())

    if voice_model is None:
        voice_model = "tts_models/en/jenny/jenny"
    if tempo is None:
        tempo = 1.5
    tts = TTS(voice_model, gpu=True)

    break_tokens = "\n"
    texts = full_text.split(break_tokens)

    subp = None
    ntf = ntf_old = None
    for text in texts:
        if not text:
            continue
        if len(text) > 250:
            i = 0
            while i < len(text):
                text2 = text[i : i + 250]
                i += 250
                ntf = tempfile.NamedTemporaryFile()
                attempts = 3
                for j in range(attempts):
                    try:
                        tts.tts_to_file(
                            text=text2,
                            # language="en",
                            # speaker_wav=dir_path
                            # + os.sep
                            # + "do-for-you-nice-female-spoken-vocal_77bpm_E_minor.wav",
                            file_path=ntf.name,
                        )
                        break
                    except RuntimeError as ree:
                        print(ree, file=sys.stderr)
                else:
                    raise RuntimeError("Exceeded TTS attempts")
                if subp is not None:
                    subp.wait()
                if ntf_old is not None:
                    ntf_old.close()
                subp = subprocess.Popen(
                    f"ffplay -nodisp -autoexit -af atempo={tempo} {ntf.name}",
                    shell=True,
                )
                ntf_old = ntf
        else:
            ntf = tempfile.NamedTemporaryFile(delete=False)
            attempts = 3
            for i in range(attempts):
                try:
                    tts.tts_to_file(
                        text=text,
                        # language="en",
                        # speaker_wav=dir_path
                        # + os.sep
                        # + "do-for-you-nice-female-spoken-vocal_77bpm_E_minor.wav",
                        file_path=ntf.name,
                    )
                    break
                except RuntimeError as ree:
                    print(ree, file=sys.stderr)
            else:
                raise RuntimeError("Exceeded TTS attempts")

            if subp is not None:
                subp.wait()
            if ntf_old is not None:
                ntf_old.close()
            subp = subprocess.Popen(
                f"ffplay -nodisp -autoexit -af atempo={tempo} {ntf.name}", shell=True
            )
            ntf_old = ntf
    if subp is not None:
        subp.wait()
    if ntf is not None:
        ntf.close()
    if ntf_old is not None:
        ntf_old.close()


if __name__ == "__main__":
    try:
        arguments = docopt(__doc__)
        tempo = float(arguments["--tempo"])
        voice_model = arguments["--voice_model"]
    except:
        #fuck off
        tempo = 1.4
        voice_model = None
    #text = arguments["<text>"]
    #print(text)

    result = subprocess.run(
        ['xclip', '-o'],
        capture_output=True,  # Python >= 3.7 only
        text=True,
    )
    text = result.stdout

    printable = set(string.printable)
    text = ''.join(filter(lambda x: x in printable, text))
    print(text)

    if text[0:2] == "$(" and text[-1] == ")":
        try:
            print("converting...")
            text = subprocess.check_output(text[2:-1], shell=True, timeout=0.1).decode()
            print(text)
        except subprocess.TimeoutExpired:
            print("timed out")  # FUCK OFF!

    main(str(text), voice_model, tempo)
