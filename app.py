"""Run: python -m streamlit run app.py

Uses training_results/best_model.pth by default. Override with the
EXPRESSION_MODEL_PATH environment variable (relative paths are app-relative).
The model must be the four-class ResNet18 state_dict saved by 03_train_evaluate.py.
"""
from __future__ import annotations

import io
import json
import os
import warnings
from pathlib import Path

CLASS_NAMES = ["분노", "슬픔", "기쁨", "중립"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def prepare_image(contents: bytes):
    """Decode an upload and return an oriented preview plus a 96x96 L face."""
    from PIL import Image, ImageOps
    if not contents or len(contents) > MAX_UPLOAD_BYTES:
        raise ValueError("10MB 이하의 JPG 또는 PNG 이미지를 업로드해 주세요.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(contents)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise ValueError("JPG 또는 PNG 형식의 이미지만 사용할 수 있습니다.")
            source.load()
            oriented = ImageOps.exif_transpose(source)
            if "A" in oriented.getbands() or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                preview = Image.alpha_composite(background, rgba).convert("RGB")
            else:
                preview = oriented.convert("RGB")
            gray = preview.convert("L").resize((96, 96), Image.Resampling.BILINEAR)
    return preview, gray


def check_metadata(model_path: Path):
    metadata_path = model_path.with_name("best_model_metadata.json")
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {"architecture": "resnet18", "class_names": CLASS_NAMES,
                "input_size": [3, 224, 224], "mean": MEAN, "std": STD}
    for key, value in expected.items():
        if key in metadata and metadata[key] != value:
            raise ValueError(f"모델의 {key} 설정이 앱과 일치하지 않습니다.")


def main():
    import streamlit as st
    import torch
    from torch import nn
    from torchvision import transforms
    from torchvision.models import resnet18
    from PIL import Image, UnidentifiedImageError

    st.set_page_config(page_title="표정 인식 AI 서비스", page_icon="🙂", layout="centered")
    st.title("내 손으로 만든 표정 인식 AI 서비스")
    st.write("얼굴 이미지를 업로드하면 분노, 슬픔, 기쁨, 중립 중 어떤 표정인지 예측합니다.")
    st.caption("한 사람의 얼굴이 크게 보이는 이미지를 사용해 주세요.")

    here = Path(__file__).resolve().parent
    model_path = Path(os.environ.get("EXPRESSION_MODEL_PATH", "training_results/best_model.pth")).expanduser()
    if not model_path.is_absolute():
        model_path = here / model_path
    model_path = model_path.resolve()

    @st.cache_resource(show_spinner="표정 인식 모델을 준비하고 있습니다…")
    def load_model(path: str, modified_ns: int, size: int, metadata_modified_ns: int):
        # File signatures invalidate cached weights when a new best model is saved.
        checkpoint_path = Path(path)
        check_metadata(checkpoint_path)
        model = resnet18(weights=None)  # Inference needs no pretrained download.
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        # CPU makes this small, single-image service portable and thread-safe for reads.
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
        return model, transform

    uploaded = st.file_uploader("얼굴 이미지 업로드", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    if not model_path.is_file():
        st.warning("학습된 모델이 아직 준비되지 않았습니다.")
        st.info("학습 코드로 생성한 best_model.pth를 앱의 training_results 폴더에 넣은 후 다시 실행해 주세요.")
        st.stop()
    try:
        signature = model_path.stat()
        metadata = model_path.with_name("best_model_metadata.json")
        model, transform = load_model(str(model_path), signature.st_mtime_ns, signature.st_size,
                                      metadata.stat().st_mtime_ns if metadata.exists() else 0)
    except Exception as exc:
        st.error("모델을 불러올 수 없습니다. 이 앱용 4개 클래스 ResNet18 가중치인지 확인해 주세요.")
        with st.expander("오류 상세"):
            st.text(str(exc))
        st.stop()

    if uploaded is None:
        st.info("JPG 또는 PNG 파일을 선택하면 예측 결과가 표시됩니다.")
        return
    try:
        preview, gray = prepare_image(uploaded.getvalue())
    except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        st.error(f"이미지를 읽을 수 없습니다: {exc}")
        st.stop()

    left, right = st.columns([2, 1])
    with left:
        st.image(preview, caption="업로드한 얼굴 이미지", use_container_width=True)
    with right:
        st.image(gray, caption="전처리 이미지 · 흑백 96×96", width=160)

    try:
        with st.spinner("표정을 분석하고 있습니다…"):
            # Training images are 96x96, then expanded to RGB/224x224 and normalized.
            tensor = transform(gray).unsqueeze(0)
            with torch.inference_mode():
                logits = model(tensor)
                probabilities = torch.softmax(logits, dim=1)[0]
            if probabilities.shape != (4,) or not torch.isfinite(probabilities).all():
                raise ValueError("모델의 확률 출력이 올바르지 않습니다.")
            values = probabilities.tolist()
            predicted = int(probabilities.argmax().item())
    except Exception as exc:
        st.error("표정 예측 중 오류가 발생했습니다.")
        with st.expander("오류 상세"):
            st.text(str(exc))
        st.stop()

    st.success(f"예측된 표정: **{CLASS_NAMES[predicted]}**")
    st.metric("Confidence Score", f"{values[predicted] * 100:.2f}%")
    st.subheader("감정별 예측 확률")
    st.bar_chart({"감정": CLASS_NAMES, "예측 확률 (%)": [v * 100 for v in values]},
                 x="감정", y="예측 확률 (%)")
    columns = st.columns(4)
    for column, name, probability in zip(columns, CLASS_NAMES, values):
        column.metric(name, f"{probability * 100:.2f}%")


if __name__ == "__main__":
    main()
