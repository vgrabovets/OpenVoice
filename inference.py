import tempfile
from collections import defaultdict
from typing import Literal

import torch
from melo.api import TTS
from path import Path
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from openvoice.logger import logger

CONVERTER_PATH = 'checkpoints_v2/converter'

LANGUAGE_CODES = {
    'en': 'EN_NEWEST',
    'ja': 'JP'
}


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference_speaker_path: str
    save_path: str
    language: Literal['en', 'ja']
    text: str


def generate_audio(requests: list[GenerateRequest]) -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f'Using {device}')

    tone_color_converter = ToneColorConverter(
        f'{CONVERTER_PATH}/config.json',
        device=device
    )
    tone_color_converter.load_ckpt(f'{CONVERTER_PATH}/checkpoint.pth')
    requests_by_language = defaultdict(list)

    for request in requests:
        save_path = Path(request.save_path)
        if save_path.splitext()[-1] != '.wav':
            logger.error(f'{request.save_path} should have extension .wav')
            return

        language_code = LANGUAGE_CODES[request.language]
        requests_by_language[language_code].append(request)

    speakers = {}
    for language, requests_list in requests_by_language.items():
        model = TTS(language=language, device=device)
        speaker_se_key = language.lower().replace('_', '-')
        speaker_se = torch.load(
            f'checkpoints_v2/base_speakers/ses/{speaker_se_key}.pth',
            map_location=device
        )
        if len(model.hps.data.spk2id) > 1:
            logger.warning(f'There are several speaker ids: {model.hps.data.spk2id}')

        logger.info(f'Processing language: {language}')
        for request in tqdm(requests_list):
            if request.reference_speaker_path not in speakers:
                logger.info(
                    f'Getting tone color embedding for {request.reference_speaker_path}'
                )
                speakers[request.reference_speaker_path], _ = se_extractor.get_se(
                    request.reference_speaker_path,
                    tone_color_converter,
                    vad=False
                )

            save_path = Path(request.save_path)
            temp_audio_save_path = Path(tempfile.gettempdir()) / save_path.name
            model.tts_to_file(
                request.text,
                speaker_id=0,
                output_path=temp_audio_save_path,
                speed=1,
                sdp_ratio=0.5,
                quiet=True,
            )

            tone_color_converter.convert(
                audio_src_path=temp_audio_save_path,
                src_se=speaker_se,
                tgt_se=speakers[request.reference_speaker_path],
                output_path=save_path,
                tau=0.5,
            )
