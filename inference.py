import tempfile
from collections import defaultdict
from typing import Annotated, Literal, Optional

import torch
from annotated_types import MinLen
from melo.api import TTS
from path import Path
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from tqdm import tqdm

from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from openvoice.logger import logger

class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference_speaker_path: str
    speaker_embedding_path: Optional[str] = None
    save_path: str
    language: Literal['en', 'ja']
    text: Annotated[str, MinLen(1)]


class Generator:
    _converter_path = 'checkpoints_v2/converter'
    _language_codes = {
        'en': 'EN_NEWEST',
        'ja': 'JP'
    }

    def __init__(self):
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f'Device: {self._device}')

        self._tone_color_converter = None
        self._models: dict[str, TTS] = {}
        self._base_speakers: dict[str, Tensor] = {}
        self._reference_speakers: dict[str, Tensor] = {}

    def generate(self, requests: list[GenerateRequest]) -> None:
        requests_by_language = defaultdict(list)
        for request in requests:
            save_path = Path(request.save_path)
            if save_path.splitext()[-1] != '.wav':
                logger.error(f'{request.save_path} should have extension .wav')
                return

            language_code = self._language_codes[request.language]
            requests_by_language[language_code].append(request)

        for language, requests_list in requests_by_language.items():
            logger.info(f'Processing language: {language}')
            self._load_model(language)

            for request in tqdm(requests_list):
                request: GenerateRequest
                self._load_reference_speaker(
                    reference_speaker_audio_path=request.reference_speaker_path,
                    speaker_embedding_path=request.speaker_embedding_path
                )
                save_path = Path(request.save_path)
                pronunciation_base_speaker_path = Path(
                    tempfile.gettempdir()
                ) / save_path.name
                self._models[language].tts_to_file(
                    request.text,
                    speaker_id=0,
                    output_path=pronunciation_base_speaker_path,
                    speed=1,
                    sdp_ratio=0.5,
                    quiet=True,
                )

                self._tone_color_converter.convert(
                    audio_src_path=pronunciation_base_speaker_path,
                    src_se=self._base_speakers[language],
                    tgt_se=self._reference_speakers[request.reference_speaker_path],
                    output_path=save_path,
                    tau=0.3,
                )

    def _load_model(self, language: str) -> None:
        if self._tone_color_converter is None:
            logger.info('Loading tone color converter')
            self._tone_color_converter = ToneColorConverter(
                f'{self._converter_path}/config.json',
                device=self._device
            )
            self._tone_color_converter.load_ckpt(f'{self._converter_path}/checkpoint.pth')

        if language not in self._models:
            logger.info(f'Loading TTS model for {language}')
            self._models[language] = TTS(language=language, device=self._device)
            if len(self._models[language].hps.data.spk2id) > 1:
                logger.warning(f'There are several speaker ids: {self._models[language].hps.data.spk2id}')

        if language not in self._base_speakers:
            logger.info(f'Loading speaker for {language}')
            speaker_se_key = language.lower().replace('_', '-')
            speaker_se = torch.load(
                f'checkpoints_v2/base_speakers/ses/{speaker_se_key}.pth',
                map_location=self._device
            )
            self._base_speakers[language] = speaker_se

    def _load_reference_speaker(
        self,
        reference_speaker_audio_path: str,
        speaker_embedding_path: str
    ) -> None:
        if reference_speaker_audio_path not in self._reference_speakers:
            if speaker_embedding_path:
                logger.info(
                    f'Using cached embeddings for {reference_speaker_audio_path}'
                )
                se_embedding = torch.load(
                    speaker_embedding_path,
                    map_location=self._device
                )
            else:
                logger.info(
                    f'Getting tone color embedding for {reference_speaker_audio_path}'
                )
                se_embedding, _ = se_extractor.get_se(
                    audio_path=reference_speaker_audio_path,
                    vc_model=self._tone_color_converter,
                    target_dir=f'{tempfile.gettempdir()}/open_voice',
                    vad=False
                )
                speaker_embedding_path = Path(
                    reference_speaker_audio_path
                ).parent / 'se.pth'
                torch.save(se_embedding, speaker_embedding_path)
                logger.info(f'Speaker embedding saved to {speaker_embedding_path}')

            self._reference_speakers[reference_speaker_audio_path] = se_embedding
