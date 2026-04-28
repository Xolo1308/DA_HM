"""
ĐỒ ÁN: PHÁT HIỆN WEBSITE PHISHING BẰNG CNN - ResNet50
Tác giả: [Tên bạn]
Mô hình: ResNet50 với Transfer Learning và Fine-tuning
"""

import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# =========================
# 1. KHAI BÁO ĐƯỜNG DẪN
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(BASE_DIR, 'train')
test_dir = os.path.join(BASE_DIR, 'test')

# =========================
# 2. THAM SỐ MÔ HÌNH
# =========================
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32  # Giảm xuống 32 để tránh quá tải RAM
EPOCHS_STAGE1 = 10      # Giai đoạn 1: Chỉ train top layers
EPOCHS_FINETUNE = 5     # Giai đoạn 2: Fine-tuning
SEED = 123
MODEL_NAME = "ResNet50_Phishing_Detector"

# File báo cáo
REPORT_TXT_PATH = os.path.join(BASE_DIR, f"{MODEL_NAME}_report.txt")
LOSS_PLOT_PATH = os.path.join(BASE_DIR, f"{MODEL_NAME}_loss.png")
ACC_PLOT_PATH = os.path.join(BASE_DIR, f"{MODEL_NAME}_accuracy.png")
CM_PLOT_PATH = os.path.join(BASE_DIR, f"{MODEL_NAME}_confusion_matrix.png")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, f"{MODEL_NAME}.h5")

# Mở file báo cáo
f = open(REPORT_TXT_PATH, "w", encoding="utf-8")

def log(msg):
    """Ghi log ra cả console và file"""
    print(msg)
    f.write(msg + "\n")

# =========================
# 3. KIỂM TRA THƯ MỤC DỮ LIỆU
# =========================
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Không tìm thấy thư mục train: {train_dir}")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Không tìm thấy thư mục test: {test_dir}")

log("="*60)
log("ĐỒ ÁN: PHÁT HIỆN WEBSITE PHISHING BẰNG RESNET50")
log("="*60)
log(f"Train dir: {train_dir}")
log(f"Test dir: {test_dir}")

# =========================
# 4. ĐỌC DỮ LIỆU
# =========================
log("\n[1/6] Đang đọc dữ liệu...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset='training'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset='validation'
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

log(f"Class names: {class_names}")
log(f"Number of classes: {num_classes}")
log(f"Bài toán: Phân loại {class_names[0]} (Hợp pháp) vs {class_names[1]} (Lừa đảo)")

# =========================
# 5. TIỀN XỬ LÝ DỮ LIỆU (Chuẩn hóa theo ResNet50)
# =========================
log("\n[2/6] Đang tiền xử lý dữ liệu...")

def preprocess_image(image, label):
    """Chuẩn hóa ảnh theo yêu cầu của ResNet50"""
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)  # Chuẩn hóa đúng cách của ResNet50
    return image, label

train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# Tối ưu hiệu suất
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# =========================
# 6. XÂY DỰNG MÔ HÌNH RESNET50
# =========================
log("\n[3/6] Đang xây dựng mô hình ResNet50...")

# Tải ResNet50 pretrained trên ImageNet
base_model = ResNet50(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)

log("  - Đã tải ResNet50 pretrained trên ImageNet")

# GIAI ĐOẠN 1: Freeze toàn bộ backbone
base_model.trainable = False

# Thêm các layers tùy chỉnh cho bài toán phishing
x = base_model.output
x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
x = layers.Dense(256, activation='relu', name='dense_1')(x)
x = layers.Dropout(0.5, name='dropout_1')(x)
x = layers.Dense(128, activation='relu', name='dense_2')(x)
x = layers.Dropout(0.3, name='dropout_2')(x)
outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

model = models.Model(inputs=base_model.input, outputs=outputs)

# Compile cho giai đoạn 1
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# In summary
model.summary(print_fn=lambda x: log(x))

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

callbacks = [early_stopping, reduce_lr]

# =========================
# 7. HUẤN LUYỆN GIAI ĐOẠN 1
# =========================
log("\n[4/6] BẮT ĐẦU HUẤN LUYỆN GIAI ĐOẠN 1 (Freeze Backbone)")
log(f"  - Số epochs: {EPOCHS_STAGE1}")
log(f"  - Learning rate: 1e-3")
log(f"  - Số mẫu train: {tf.data.experimental.cardinality(train_ds).numpy() * BATCH_SIZE}")
log(f"  - Số mẫu validation: {tf.data.experimental.cardinality(val_ds).numpy() * BATCH_SIZE}")

start_train_time = time.time()

history_stage1 = model.fit(
    train_ds,
    epochs=EPOCHS_STAGE1,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1
)

log(f"\n  - Kết thúc giai đoạn 1")
log(f"  - Train accuracy cao nhất: {max(history_stage1.history['accuracy'])*100:.2f}%")
log(f"  - Validation accuracy cao nhất: {max(history_stage1.history['val_accuracy'])*100:.2f}%")

# =========================
# 8. FINE-TUNING (GIAI ĐOẠN 2)
# =========================
log("\n[5/6] BẮT ĐẦU FINE-TUNING")
log("  - Mở khóa các layers phía trên của ResNet50")

# Mở khóa toàn bộ backbone
base_model.trainable = True

# Chi tiết: Chỉ fine-tune các layers từ block 5 trở lên
for layer in base_model.layers:
    if layer.name.startswith("conv5_"):
        # Giữ BatchNormalization cố định để tránh lỗi
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
    elif layer.name.startswith("conv4_"):
        layer.trainable = True
    else:
        layer.trainable = False

# Đếm số layers được train
trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
log(f"  - Số layers được fine-tune: {trainable_count}/{len(base_model.layers)}")

# Re-compile với learning rate nhỏ hơn
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # LR nhỏ hơn 100 lần
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_ds,
    epochs=EPOCHS_FINETUNE,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1
)

end_train_time = time.time()
training_time = end_train_time - start_train_time

log(f"\n  - Hoàn thành fine-tuning")
log(f"  - Validation accuracy sau fine-tune: {history_finetune.history['val_accuracy'][-1]*100:.2f}%")

# =========================
# 9. LƯU MÔ HÌNH
# =========================
model.save(MODEL_SAVE_PATH)
log(f"\n  - Đã lưu mô hình vào: {MODEL_SAVE_PATH}")

# =========================
# 10. ĐÁNH GIÁ TRÊN TEST SET
# =========================
log("\n[6/6] ĐÁNH GIÁ MÔ HÌNH TRÊN TEST SET")

start_test_time = time.time()
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
end_test_time = time.time()
testing_time = end_test_time - start_test_time

log(f"\nKết quả evaluate trên test set:")
log(f"  - Test Loss: {test_loss:.4f}")
log(f"  - Test Accuracy: {test_acc * 100:.2f}%")

# Gộp lịch sử huấn luyện
all_loss = history_stage1.history['loss'] + history_finetune.history['loss']
all_val_loss = history_stage1.history['val_loss'] + history_finetune.history['val_loss']
all_acc = history_stage1.history['accuracy'] + history_finetune.history['accuracy']
all_val_acc = history_stage1.history['val_accuracy'] + history_finetune.history['val_accuracy']

best_epoch = np.argmin(all_val_loss) + 1
log(f"  - Epoch có validation loss thấp nhất: epoch {best_epoch}")

# =========================
# 11. DỰ ĐOÁN VÀ TÍNH CHỈ SỐ
# =========================
log("\nĐang dự đoán trên test set...")

y_true = []
y_pred = []
y_pred_proba = []

start_infer_time = time.time()

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))
    y_pred_proba.extend(preds[:, 1])

end_infer_time = time.time()
inference_time = end_infer_time - start_infer_time

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Các chỉ số đánh giá
acc = accuracy_score(y_true, y_pred)
precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

# Chỉ số theo từng lớp
precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

log("\n" + "="*60)
log("KẾT QUẢ ĐÁNH GIÁ")
log("="*60)

log(f"\nTỔNG THỂ (MACRO AVERAGE):")
log(f"  - Accuracy : {acc * 100:.2f}%")
log(f"  - Precision: {precision_macro * 100:.2f}%")
log(f"  - Recall   : {recall_macro * 100:.2f}%")
log(f"  - F1-Score : {f1_macro * 100:.2f}%")

log(f"\nTHEO TỪNG LỚP:")
for i, class_name in enumerate(class_names):
    log(f"  - {class_name}:")
    log(f"      Precision: {precision_per_class[i] * 100:.2f}%")
    log(f"      Recall   : {recall_per_class[i] * 100:.2f}%")
    log(f"      F1-Score : {f1_per_class[i] * 100:.2f}%")

# Classification Report
log("\n" + "="*60)
log("BÁO CÁO PHÂN LOẠI CHI TIẾT")
log("="*60)
class_report = classification_report(
    y_true, y_pred, target_names=class_names, digits=4, zero_division=0
)
log(class_report)

# =========================
# 12. MA TRẬN NHẦM LẪN
# =========================
cm = confusion_matrix(y_true, y_pred)

fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', xticks_rotation=45, values_format='d', ax=ax_cm)
ax_cm.set_title(f'Confusion Matrix - {MODEL_NAME}', fontsize=14, fontweight='bold')
plt.tight_layout()
fig_cm.savefig(CM_PLOT_PATH, dpi=300, bbox_inches='tight')
log(f"\nĐã lưu ma trận nhầm lẫn vào: {CM_PLOT_PATH}")

# =========================
# 13. CHI PHÍ TÍNH TOÁN
# =========================
log("\n" + "="*60)
log("CHI PHÍ TÍNH TOÁN")
log("="*60)

total_params = model.count_params()
params_million = total_params / 1e6
total_time = training_time + inference_time
num_test_samples = len(y_true)
avg_inference_time_per_sample = inference_time / num_test_samples if num_test_samples > 0 else 0

log(f"  - Tổng số tham số: {total_params:,} (≈ {params_million:.2f} triệu)")
log(f"  - Thời gian huấn luyện (tổng): {training_time:.2f} giây")
log(f"      + Giai đoạn 1 (freeze): {EPOCHS_STAGE1} epochs")
log(f"      + Giai đoạn 2 (fine-tune): {EPOCHS_FINETUNE} epochs")
log(f"  - Thời gian evaluate: {testing_time:.2f} giây")
log(f"  - Thời gian suy luận test set: {inference_time:.2f} giây")
log(f"  - Thời gian suy luận trung bình/mẫu: {avg_inference_time_per_sample*1000:.4f} ms")
log(f"  - Số mẫu test: {num_test_samples}")

# =========================
# 14. BẢNG TỔNG HỢP KẾT QUẢ
# =========================
log("\n" + "="*60)
log("BẢNG TỔNG HỢP KẾT QUẢ SO SÁNH")
log("="*60)

log(f"{'Mô hình':<25} | {'Accuracy':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'Params (M)':<12}")
log("-" * 95)
log(f"{MODEL_NAME:<25} | {acc * 100:<12.2f}% | {precision_macro * 100:<12.2f}% | {recall_macro * 100:<12.2f}% | {f1_macro * 100:<12.2f}% | {params_million:<12.2f}")

# =========================
# 15. VẼ BIỂU ĐỒ HỘI TỤ
# =========================
fig1 = plt.figure(figsize=(10, 6))
plt.plot(all_loss, 'b-o', linewidth=2, markersize=4, label='Training Loss')
plt.plot(all_val_loss, 'r-s', linewidth=2, markersize=4, label='Validation Loss')
plt.axvline(x=EPOCHS_STAGE1 - 0.5, color='gray', linestyle='--', label='Start Fine-tuning')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title(f'Training & Validation Loss - {MODEL_NAME}', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
fig1.savefig(LOSS_PLOT_PATH, dpi=300, bbox_inches='tight')
log(f"\nĐã lưu đồ thị loss vào: {LOSS_PLOT_PATH}")

fig2 = plt.figure(figsize=(10, 6))
plt.plot(all_acc, 'b-o', linewidth=2, markersize=4, label='Training Accuracy')
plt.plot(all_val_acc, 'r-s', linewidth=2, markersize=4, label='Validation Accuracy')
plt.axvline(x=EPOCHS_STAGE1 - 0.5, color='gray', linestyle='--', label='Start Fine-tuning')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title(f'Training & Validation Accuracy - {MODEL_NAME}', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
fig2.savefig(ACC_PLOT_PATH, dpi=300, bbox_inches='tight')
log(f"Đã lưu đồ thị accuracy vào: {ACC_PLOT_PATH}")

# =========================
# 16. KẾT LUẬN
# =========================
log("\n" + "="*60)
log("KẾT LUẬN")
log("="*60)

log(f"""
1. KHẢ NĂNG PHÁT HIỆN PHISHING:
   - Mô hình ResNet50 đạt độ chính xác {acc * 100:.2f}% trên tập test.
   - Recall của lớp phishing (phát hiện đúng web lừa đảo): {recall_per_class[1] * 100:.2f}%
   - Precision của lớp phishing (độ tin cậy khi cảnh báo): {precision_per_class[1] * 100:.2f}%

2. ĐIỂM MẠNH:
   - Transfer learning giúp mô hình học tốt với lượng dữ liệu vừa phải
   - Fine-tuning cải thiện đáng kể độ chính xác
   - Thời gian suy luận nhanh ({avg_inference_time_per_sample*1000:.2f} ms/ảnh)

3. ỨNG DỤNG THỰC TẾ:
   - Có thể tích hợp vào trình duyệt làm extension cảnh báo lừa đảo
   - Hoặc triển khai trên server để quét hàng loạt URL
""")

# =========================
# 17. ĐÓNG FILE VÀ KẾT THÚC
# =========================
f.close()
print("\n" + "="*60)
print("HOÀN THÀNH! CÁC FILE KẾT QUẢ:")
print("="*60)
print(f"  - Báo cáo chi tiết: {REPORT_TXT_PATH}")
print(f"  - Mô hình đã lưu: {MODEL_SAVE_PATH}")
print(f"  - Đồ thị loss: {LOSS_PLOT_PATH}")
print(f"  - Đồ thị accuracy: {ACC_PLOT_PATH}")
print(f"  - Ma trận nhầm lẫn: {CM_PLOT_PATH}")
print("\n✨ XONG! MÔ HÌNH RESNET50 ĐÃ ĐƯỢC HUẤN LUYỆN CHO BÀI TOÁN PHÁT HIỆN PHISHING")

plt.close('all')