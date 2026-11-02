# Nguồn cần trích dẫn

## Neural TTS

- Model: `facebook/mms-tts-vie`
- Model family: Massively Multilingual Speech (MMS), Meta AI
- License: CC BY-NC 4.0
- Model card: https://huggingface.co/facebook/mms-tts-vie
- Paper: Vineel Pratap et al., “Scaling Speech Technology to 1,000+ Languages”, 2023.

## Tài liệu của Lab

- X. Huang, A. Acero, H.-W. Hon, *Spoken Language Processing*, Prentice Hall, 2001, Chapters 14-16.
- L. R. Rabiner, R. W. Schafer, *Theory and Applications of Digital Speech Processing*, Prentice Hall, 2010, Chapter 13.

## Inventory

Inventory được tạo bằng giọng tổng hợp `vi-VN-HoaiMyNeural` qua Edge TTS, sau đó chuyển về mono 16 kHz PCM 16-bit, trim silence và cân RMS có kiểm soát. Không mô tả inventory này là bản thu trực tiếp của con người.

## Automated intelligibility proxy

- Model: `vinai/PhoWhisper-tiny`
- License: BSD 3-Clause
- Model card: https://huggingface.co/vinai/PhoWhisper-tiny
- Thanh-Thien Le, Linh The Nguyen, Dat Quoc Nguyen, “PhoWhisper: Automatic Speech Recognition for Vietnamese”, ICLR 2024 Tiny Papers.
