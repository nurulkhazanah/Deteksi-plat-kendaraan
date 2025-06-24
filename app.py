import gradio as gr
import torch
import os
from PIL import Image, ImageDraw
import numpy as np
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
import easyocr
import re  # Import regex for advanced text parsing

# ------------------- Model and Data Loading Functions (from notebook) -------------------


class CarPlateDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df,
        image_dir,
        unique_img,
        indices,
        transform=None,
        test_mode=False,
        val_mode=False,
    ):
        self.df = df
        self.image_dir = image_dir
        self.unique_img = unique_img
        self.indices = indices
        self.transform = transform
        self.test_mode = test_mode
        self.val_mode = val_mode

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        pass


def custom_collate(data):
    return data


def load_frcnn():
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 2  # 1 for license plate, 1 for background
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_channels=in_features, num_classes=num_classes
    )

    return model


# ------------------- OCR Setup -------------------
reader = easyocr.Reader(
    ["en"]
)  # 'en' for English characters, consider adding 'id' if EasyOCR supports it well for this use case

# ------------------- Load Model -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_frcnn()
model.load_state_dict(torch.load("frcnn.pth", map_location=device))
model.to(device)
model.eval()  # Set model to evaluation mode

# ------------------- Data Plat Nomor Indonesia (Extracted from Auto2000 link) -------------------
# Prioritaskan prefiks yang lebih panjang dulu (misal: AD sebelum A) untuk pencocokan yang lebih akurat
LICENSE_PLATE_PREFIXES = {
    "AD": "Jawa Tengah (Surakarta/Solo, Sukoharjo, Klaten, Boyolali, Karanganyar, Sragen, Wonogiri)",
    "AA": "Jawa Tengah (Kedu: Magelang, Temanggung, Wonosobo, Kebumen, Purworejo)",
    "AB": "Daerah Istimewa Yogyakarta",
    "AG": "Jawa Timur (Kediri, Tulungagung, Blitar, Trenggalek, Nganjuk)",
    "AE": "Jawa Timur (Madiun, Ngawi, Pacitan, Ponorogo, Magetan)",
    "DK": "Bali",
    "DR": "Nusa Tenggara Barat (Lombok)",
    "EA": "Nusa Tenggara Barat (Sumbawa)",
    "ED": "Nusa Tenggara Timur (Sumba)",
    "EB": "Nusa Tenggara Timur (Flores)",
    "DH": "Nusa Tenggara Timur (Timor)",
    "KB": "Kalimantan Barat",
    "DA": "Kalimantan Selatan",
    "KH": "Kalimantan Tengah",
    "KT": "Kalimantan Timur",
    "KU": "Kalimantan Utara",
    "DB": "Sulawesi Utara (Manado, Minahasa Utara, Bitung, Tomohon, Kotamobagu)",
    "DL": "Sulawesi Utara (Sitaro, Talaud, Sangihe)",
    "DM": "Gorontalo",
    "DN": "Sulawesi Tengah",
    "DT": "Sulawesi Tenggara",
    "DD": "Sulawesi Selatan",
    "DC": "Sulawesi Barat",
    "PA": "Papua (termasuk Papua Barat)",
    "DE": "Maluku (Ambon)",
    "DG": "Maluku Utara (Ternate, Tidore)",
    "BL": "Aceh",
    "BB": "Sumatera Utara (Barat)",
    "BK": "Sumatera Utara (Timur)",
    "BA": "Sumatera Barat",
    "BD": "Bengkulu",
    "BE": "Lampung",
    "BG": "Sumatera Selatan",
    "BH": "Jambi",
    "BP": "Kepulauan Riau (Batam, Bintan, Tanjungpinang, Karimun, Lingga, Anambas, Natuna)",
    "BN": "Kepulauan Bangka Belitung",
    "A": "Banten (Serang, Cilegon, Lebak, Pandeglang)",
    "B": "DKI Jakarta, Bekasi, Tangerang",
    "D": "Jawa Barat (Bandung, Cimahi, Bandung Barat)",
    "E": "Jawa Barat (Cirebon, Indramayu, Majalengka, Kuningan)",
    "F": "Jawa Barat (Bogor, Sukabumi, Cianjur)",
    "G": "Jawa Tengah (Pekalongan, Tegal, Batang, Brebes)",
    "H": "Jawa Tengah (Semarang, Salatiga, Kendal, Demak)",
    "K": "Jawa Tengah (Pati, Kudus, Jepara, Rembang, Blora, Grobogan)",
    "L": "Jawa Timur (Surabaya)",
    "M": "Jawa Timur (Madura: Pamekasan, Sampang, Sumenep, Bangkalan)",
    "N": "Jawa Timur (Malang, Pasuruan, Probolinggo, Lumajang, Batu)",
    "P": "Jawa Timur (Besuki: Bondowoso, Situbondo, Jember, Banyuwangi)",
    "S": "Jawa Timur (Lamongan, Tuban, Bojonegoro, Mojokerto, Jombang)",
    "T": "Jawa Barat (Karawang, Purwakarta, Subang)",
    "R": "Jawa Tengah (Banyumas, Cilacap, Purbalingga, Banjarnegara)",
    # 'Z': 'Jawa Barat (Garut, Tasikmalaya, Sumedang, Ciamis, Banjar)', # Example from link, not explicitly mentioned how it relates to common prefixes
    # 'DXXX': 'Contoh plat khusus/dinas, tidak selalu terdeteksi dengan pola huruf awal'
}


# Fungsi pembantu untuk mengekstrak prefiks plat nomor yang relevan
def extract_plate_prefix(ocr_text):
    # Membersihkan teks dari spasi berlebih dan karakter non-alfanumerik yang tidak relevan
    cleaned_text = re.sub(r"[^A-Z0-9]", "", ocr_text.upper())

    # Mencari pola: huruf-huruf di awal, sebelum angka pertama
    match = re.match(r"([A-Z]+)", cleaned_text)
    if match:
        potential_prefix = match.group(1)

        # Prioritaskan pencarian prefiks yang lebih panjang (2 karakter)
        for prefix in sorted(LICENSE_PLATE_PREFIXES.keys(), key=len, reverse=True):
            if potential_prefix.startswith(prefix):
                return prefix
    return None


# ------------------- Prediction Function -------------------


def predict_plate(input_image: Image.Image):
    """
    Function to detect car plates, extract text using FRCNN and OCR, and identify the region.
    Input: PIL Image
    Output: PIL Image with bounding boxes, extracted text string, and detected region.
    """
    img_tensor = transforms.ToTensor()(input_image).to(device)

    with torch.no_grad():
        output = model([img_tensor])

    out_boxes = output[0]["boxes"]
    out_scores = output[0]["scores"]

    # Perform NMS to filter out some of the bounding boxes
    keep = nms(out_boxes, out_scores, 0.45)  # NMS threshold can be adjusted
    out_boxes = out_boxes[keep]
    out_scores = out_scores[keep]

    # Convert tensor image back to PIL for drawing and cropping
    img_display = (img_tensor.permute(1, 2, 0).cpu().detach().numpy() * 255).astype(
        "uint8"
    )
    img_display = Image.fromarray(img_display)
    draw = ImageDraw.Draw(img_display)

    extracted_texts = []
    detected_regions = []

    if out_boxes.shape[0] > 0:
        for i, box in enumerate(out_boxes):
            x1, y1, x2, y2 = [int(b) for b in box]
            draw.rectangle([x1, y1, x2, y2], fill=None, outline="red", width=3)

            # Crop the detected license plate region
            cropped_plate = input_image.crop((x1, y1, x2, y2))

            # Perform OCR on the cropped plate
            cropped_plate_np = np.array(cropped_plate)

            current_plate_text = ""
            current_region = "Tidak diketahui"

            try:
                # result is a list of tuples: (bbox, text, prob)
                result = reader.readtext(cropped_plate_np)

                # We assume the most prominent text is the plate number
                if result:
                    # Sort results by confidence or by position (top-left first)
                    # For a single line plate, usually the first result is the main one.
                    main_text = result[0][
                        1
                    ]  # Get the text from the first detected result
                    main_confidence = result[0][2]

                    extracted_texts.append(
                        f"Plat {i+1} Teks: '{main_text}' (Kepercayaan: {main_confidence:.2f})"
                    )

                    # Extract prefix using the helper function
                    prefix = extract_plate_prefix(main_text)
                    if prefix:
                        current_region = LICENSE_PLATE_PREFIXES.get(
                            prefix, "Tidak diketahui"
                        )

                    detected_regions.append(f"Plat {i+1} Daerah: {current_region}")

                else:
                    extracted_texts.append(
                        f"Plat {i+1} Teks: Tidak ada teks terdeteksi."
                    )
                    detected_regions.append(
                        f"Plat {i+1} Daerah: Tidak dapat ditentukan (Tidak ada teks)"
                    )

            except Exception as e:
                extracted_texts.append(f"Plat {i+1} OCR Error: {e}")
                detected_regions.append(
                    f"Plat {i+1} Daerah: Tidak dapat ditentukan (OCR Error)"
                )

    else:
        extracted_texts.append("Tidak ada plat nomor terdeteksi.")
        detected_regions.append("Tidak ada daerah terdeteksi.")

    return img_display, "\n".join(extracted_texts), "\n".join(detected_regions)


# ------------------- Gradio Interface -------------------

title = "Deteksi Plat Nomor, OCR, dan Identifikasi Daerah"
description = """
Unggah gambar mobil untuk mendeteksi plat nomor, mengekstrak teks menggunakan Faster R-CNN dan EasyOCR,
serta mengidentifikasi daerah asal plat nomor berdasarkan prefiks huruf di awal plat.
"""
article = "Model yang digunakan: Faster R-CNN dengan ResNet50-FPN backbone, dilatih untuk mendeteksi kelas 'License_Plate'. OCR dilakukan oleh EasyOCR. Data daerah plat nomor bersumber dari Auto2000 (diperbarui terakhir 24 Juni 2025)."

iface = gr.Interface(
    fn=predict_plate,
    inputs=gr.Image(type="pil", label="Upload Gambar Mobil"),
    outputs=[
        gr.Image(type="pil", label="Plat Nomor Terdeteksi"),
        gr.Textbox(label="Teks Plat Nomor Terekstraksi"),
        gr.Textbox(label="Daerah Asal Plat Nomor"),  # Output untuk daerah
    ],
    title=title,
    description=description,
    article=article,
)

iface.launch()
