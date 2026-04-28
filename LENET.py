

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================
# GIAI ĐOẠN 1: CẤU HÌNH & KHAI BÁO ĐƯỜNG DẪN
# ============================================

# Tắt các cảnh báo không cần thiết
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Đường dẫn thư mục (code tự động nhận diện)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(BASE_DIR, 'train')
test_dir = os.path.join(BASE_DIR, 'test')

# Kiểm tra thư mục dữ liệu
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Không tìm thấy thư mục train: {train_dir}")
if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Không tìm thấy thư mục test: {test_dir}")

# ============================================
# GIAI ĐOẠN 2: THAM SỐ CẤU HÌNH MÔ HÌNH
# ============================================

# Tham số dữ liệu
IMG_HEIGHT = 64          # Chiều cao ảnh
IMG_WIDTH = 64           # Chiều rộng ảnh
BATCH_SIZE = 32          # Số ảnh mỗi lần huấn luyện
NUM_CHANNELS = 3         # Ảnh màu RGB

# Tham số huấn luyện
EPOCHS = 20              # Số vòng lặp huấn luyện (tăng lên để học tốt hơn)
SEED = 42                # Khởi tạo ngẫu nhiên cố định
VALIDATION_SPLIT = 0.2   # 20% dữ liệu train dùng để validation

# Tham số mô hình
MODEL_NAME = "LeNet-5_Phishing_Detector"

# Tên file output
REPORT_TXT_PATH = os.path.join(BASE_DIR, f"{MODEL_NAME}_report.txt")
LOSS_PLOT_PATH = os.path.join(BASE_DIR, f"{MODEL_NAME}_loss.png")
ACC_PLOT_PATH = os.path.join(BASE_DIR, f"{MODEL_NAME}_accuracy.png")
CM_PLOT_PATH = os.path.join(BASE_DIR, f"{MODEL_NAME}_confusion_matrix.png")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, f"{MODEL_NAME}.h5")

# Mở file báo cáo
report_file = open(REPORT_TXT_PATH, "w", encoding="utf-8")

def log_to_file_and_console(message):
    """Ghi log ra cả console và file báo cáo"""
    print(message)
    report_file.write(message + "\n")

# ============================================
# GIAI ĐOẠN 3: ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU
# ============================================

log_to_file_and_console("=" * 60)
log_to_file_and_console("ĐỒ ÁN: PHÁT HIỆN WEBSITE PHISHING BẰNG CNN")
log_to_file_and_console("=" * 60)
log_to_file_and_console(f"Thư mục train: {train_dir}")
log_to_file_and_console(f"Thư mục test: {test_dir}")

# Đọc dữ liệu train (tự động tách validation)
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True,
    seed=SEED,
    validation_split=VALIDATION_SPLIT,
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
    validation_split=VALIDATION_SPLIT,
    subset='validation'
)

# Đọc dữ liệu test
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=False
)

# Lấy tên các lớp (class names)
class_names = train_ds.class_names
num_classes = len(class_names)

log_to_file_and_console(f"\nCác lớp dữ liệu: {class_names}")
log_to_file_and_console(f"Số lượng lớp: {num_classes}")
log_to_file_and_console(f"Loại bài toán: Phân loại ảnh website {'/'.join(class_names)}")

# ============================================
# GIAI ĐOẠN 4: AUGMENTATION & CHUẨN HÓA DỮ LIỆU
# ============================================

# Data Augmentation (Tăng cường dữ liệu - giúp chống overfitting)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal", seed=SEED),   # Lật ngang ảnh
    layers.RandomRotation(0.1, seed=SEED),        # Xoay ảnh 10%
    layers.RandomZoom(0.1, seed=SEED),            # Zoom ảnh 10%
], name="data_augmentation")

# Chuẩn hóa giá trị pixel về [0,1]
normalization_layer = layers.Rescaling(1.0 / 255)

# Áp dụng augmentation chỉ cho train set (không áp dụng cho validation/test)
def augment_and_normalize(image, label):
    image = data_augmentation(image, training=True)
    image = normalization_layer(image)
    return image, label

def normalize_only(image, label):
    image = normalization_layer(image)
    return image, label

train_ds = train_ds.map(augment_and_normalize, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(normalize_only, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(normalize_only, num_parallel_calls=tf.data.AUTOTUNE)

# Tối ưu hiệu suất đọc dữ liệu
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# ============================================
# GIAI ĐOẠN 5: XÂY DỰNG MÔ HÌNH LeNet-5
# ============================================

log_to_file_and_console("\n" + "=" * 60)
log_to_file_and_console("XÂY DỰNG MÔ HÌNH LeNet-5 CHO PHÂN LOẠI PHISHING")
log_to_file_and_console("=" * 60)

def build_lenet_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), num_classes=2):
    """
    Xây dựng mô hình LeNet-5 được tùy chỉnh cho bài toán phishing detection
    
    Kiến trúc:
    - Conv2D (6 filters, 5x5, sigmoid) + AveragePooling
    - Conv2D (16 filters, 5x5, sigmoid) + AveragePooling  
    - Flatten
    - Dense (120, sigmoid)
    - Dense (84, sigmoid)
    - Dense (num_classes, softmax)
    """
    model = models.Sequential(name="LeNet5_Phishing_Detector")
    
    # Block 1
    model.add(layers.Conv2D(6, (5, 5), padding='same', activation='sigmoid', name='conv1'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=2, name='pool1'))
    
    # Block 2
    model.add(layers.Conv2D(16, (5, 5), activation='sigmoid', name='conv2'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=2, name='pool2'))
    
    # Flatten và Fully Connected layers
    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dense(120, activation='sigmoid', name='fc1'))
    model.add(layers.Dropout(0.3, name='dropout1'))  # Thêm dropout để chống overfitting
    model.add(layers.Dense(84, activation='sigmoid', name='fc2'))
    model.add(layers.Dropout(0.3, name='dropout2'))
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))
    
    return model

# Tạo mô hình
model = build_lenet_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), num_classes=num_classes)

# Compile mô hình
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# In summary mô hình
model.summary(print_fn=lambda x: log_to_file_and_console(x))

# ============================================
# GIAI ĐOẠN 6: CALLBACKS (THEO DÕI QUÁ TRÌNH HỌC)
# ============================================

# EarlyStopping: Dừng sớm nếu validation loss không cải thiện
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# ReduceLROnPlateau: Giảm learning rate khi validation loss plateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

callbacks = [early_stopping, reduce_lr]

# ============================================
# GIAI ĐOẠN 7: HUẤN LUYỆN MÔ HÌNH
# ============================================

log_to_file_and_console("\n" + "=" * 60)
log_to_file_and_console("BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH")
log_to_file_and_console(f"Số epochs: {EPOCHS}")
log_to_file_and_console(f"Batch size: {BATCH_SIZE}")
log_to_file_and_console("=" * 60)

start_train_time = time.time()

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks,
    verbose=1
)

end_train_time = time.time()
training_time = end_train_time - start_train_time

log_to_file_and_console(f"\nThời gian huấn luyện: {training_time:.2f} giây")

# ============================================
# GIAI ĐOẠN 8: LƯU MÔ HÌNH
# ============================================

model.save(MODEL_SAVE_PATH)
log_to_file_and_console(f"\nĐã lưu mô hình vào: {MODEL_SAVE_PATH}")

# ============================================
# GIAI ĐOẠN 9: ĐÁNH GIÁ TRÊN TEST SET
# ============================================

log_to_file_and_console("\n" + "=" * 60)
log_to_file_and_console("ĐÁNH GIÁ MÔ HÌNH TRÊN TEST SET")
log_to_file_and_console("=" * 60)

start_test_time = time.time()
test_loss, test_acc = model.evaluate(test_ds, verbose=1)
end_test_time = time.time()
testing_time = end_test_time - start_test_time

log_to_file_and_console(f"\nKết quả evaluate trên test set:")
log_to_file_and_console(f"  - Test Loss: {test_loss:.4f}")
log_to_file_and_console(f"  - Test Accuracy: {test_acc * 100:.2f}%")

# Epoch có validation loss thấp nhất
best_epoch = np.argmin(history.history['val_loss']) + 1
log_to_file_and_console(f"  - Epoch có val_loss thấp nhất: epoch {best_epoch}")

# ============================================
# GIAI ĐOẠN 10: DỰ ĐOÁN & TÍNH CÁC CHỈ SỐ
# ============================================

log_to_file_and_console("\n" + "=" * 60)
log_to_file_and_console("TÍNH TOÁN CÁC CHỈ SỐ ĐÁNH GIÁ")
log_to_file_and_console("=" * 60)

# Dự đoán trên test set
y_true = []
y_pred = []
y_pred_proba = []  # Lưu xác suất dự đoán

start_infer_time = time.time()

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))
    y_pred_proba.extend(preds[:, 1])  # Xác suất thuộc lớp phishing (class 1)

end_infer_time = time.time()
inference_time = end_infer_time - start_infer_time

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Các chỉ số cơ bản
acc = accuracy_score(y_true, y_pred)

# Macro average (trung bình cộng của các lớp)
precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

# Weighted average (tính theo tỷ lệ mẫu mỗi lớp)
precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

# Per-class metrics (cho từng lớp)
precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

log_to_file_and_console("\nKẾT QUẢ ĐÁNH GIÁ TỔNG THỂ:")
log_to_file_and_console(f"  - Accuracy: {acc * 100:.2f}%")
log_to_file_and_console(f"  - Precision (macro): {precision_macro * 100:.2f}%")
log_to_file_and_console(f"  - Recall (macro): {recall_macro * 100:.2f}%")
log_to_file_and_console(f"  - F1-Score (macro): {f1_macro * 100:.2f}%")

log_to_file_and_console("\nKẾT QUẢ THEO TỪNG LỚP:")
for i, class_name in enumerate(class_names):
    log_to_file_and_console(f"  - {class_name}:")
    log_to_file_and_console(f"      Precision: {precision_per_class[i] * 100:.2f}%")
    log_to_file_and_console(f"      Recall: {recall_per_class[i] * 100:.2f}%")
    log_to_file_and_console(f"      F1-Score: {f1_per_class[i] * 100:.2f}%")

# Báo cáo chi tiết
log_to_file_and_console("\n" + "=" * 60)
log_to_file_and_console("BÁO CÁO PHÂN LOẠI CHI TIẾT (Classification Report)")
log_to_file_and_console("=" * 60)
class_report = classification_report(
    y_true, y_pred, target_names=class_names, digits=4, zero_division=0
)
log_to_file_and_console(class_report)

# ============================================
# GIAI ĐOẠN 11: MA TRẬN NHẦM LẪN
# ============================================

cm = confusion_matrix(y_true, y_pred)

fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', xticks_rotation=45, values_format='d', ax=ax_cm)
ax_cm.set_title(f'Confusion Matrix - {MODEL_NAME}', fontsize=14, fontweight='bold')
ax_cm.set_xlabel('Predicted Label', fontsize=12)
ax_cm.set_ylabel('True Label', fontsize=12)

# Thêm nhận xét trên ma trận
for text in ax_cm.texts:
    text.set_fontsize(12)
    text.set_fontweight('bold')

plt.tight_layout()
fig_cm.savefig(CM_PLOT_PATH, dpi=300, bbox_inches='tight')
log_to_file_and_console(f"\nĐã lưu ma trận nhầm lẫn vào: {CM_PLOT_PATH}")

# ============================================
# GIAI ĐOẠN 12: PHÂN TÍCH CHI PHÍ TÍNH TOÁN
# ============================================

log_to_file_and_console("\n" + "=" * 60)
log_to_file_and_console("CHI PHÍ TÍNH TOÁN")
log_to_file_and_console("=" * 60)

# Số lượng tham số
total_params = model.count_params()
params_million = total_params / 1e6

# Thời gian
total_time = training_time + inference_time
num_test_samples = len(y_true)
avg_inference_time_per_sample = inference_time / num_test_samples if num_test_samples > 0 else 0.0

log_to_file_and_console(f"  - Tổng số tham số: {total_params:,} (≈ {params_million:.4f} triệu)")
log_to_file_and_console(f"  - Thời gian huấn luyện: {training_time:.2f} giây")
log_to_file_and_console(f"  - Thời gian evaluate: {testing_time:.2f} giây")
log_to_file_and_console(f"  - Thời gian suy luận (toàn test set): {inference_time:.4f} giây")
log_to_file_and_console(f"  - Tổng thời gian (train + infer): {total_time:.2f} giây")
log_to_file_and_console(f"  - Thời gian suy luận trung bình/mẫu: {avg_inference_time_per_sample:.6f} giây")
log_to_file_and_console(f"  - Số mẫu test: {num_test_samples}")

# ============================================
# GIAI ĐOẠN 13: BẢNG TỔNG HỢP KẾT QUẢ
# ============================================

log_to_file_and_console("\n" + "=" * 60)
log_to_file_and_console("BẢNG TỔNG HỢP KẾT QUẢ")
log_to_file_and_console("=" * 60)

# In bảng kết quả đẹp
log_to_file_and_console(f"{'Mô hình':<25} | {'Accuracy':<12} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'Params (M)':<12}")
log_to_file_and_console("-" * 95)
log_to_file_and_console(f"{MODEL_NAME:<25} | {acc * 100:<12.2f}% | {precision_macro * 100:<12.2f}% | {recall_macro * 100:<12.2f}% | {f1_macro * 100:<12.2f}% | {params_million:<12.4f}")

# ============================================
# GIAI ĐOẠN 14: VẼ BIỂU ĐỒ HỘI TỤ (LOSS & ACCURACY)
# ============================================

# Đồ thị Loss
fig1 = plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], 'b-o', linewidth=2, markersize=6, label='Training Loss')
plt.plot(history.history['val_loss'], 'r-s', linewidth=2, markersize=6, label='Validation Loss')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title(f'Training & Validation Loss - {MODEL_NAME}', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
fig1.savefig(LOSS_PLOT_PATH, dpi=300, bbox_inches='tight')
log_to_file_and_console(f"\nĐã lưu đồ thị loss vào: {LOSS_PLOT_PATH}")

# Đồ thị Accuracy
fig2 = plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], 'b-o', linewidth=2, markersize=6, label='Training Accuracy')
plt.plot(history.history['val_accuracy'], 'r-s', linewidth=2, markersize=6, label='Validation Accuracy')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title(f'Training & Validation Accuracy - {MODEL_NAME}', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
fig2.savefig(ACC_PLOT_PATH, dpi=300, bbox_inches='tight')
log_to_file_and_console(f"Đã lưu đồ thị accuracy vào: {ACC_PLOT_PATH}")

# ============================================
# GIAI ĐOẠN 15: KẾT LUẬN & NHẬN XÉT
# ============================================

log_to_file_and_console("\n" + "=" * 60)
log_to_file_and_console("KẾT LUẬN & NHẬN XÉT")
log_to_file_and_console("=" * 60)

log_to_file_and_console("""
1. KHẢ NĂNG PHÁT HIỆN PHISHING:
   - Accuracy đạt {:.2f}% trên tập test.
   - Recall của lớp phishing: {:.2f}% (tỷ lệ phát hiện đúng website lừa đảo).
   - Đây là chỉ số quan trọng nhất vì cần phát hiện càng nhiều web lừa đảo càng tốt.

2. ĐIỂM MẠNH CỦA MÔ HÌNH:
   - Kích thước mô hình nhỏ ({} triệu tham số), phù hợp triển khai real-time.
   - Thời gian suy luận nhanh ({:.4f} giây/ảnh).
   - Không yêu cầu phần cứng mạnh (có thể chạy trên CPU).

3. ĐIỂM YẾU & HƯỚNG CẢI THIỆN:
   - Nếu accuracy chưa cao, có thể tăng kích thước ảnh lên 128x128 hoặc 224x224.
   - Sử dụng transfer learning (VGG16, ResNet50) để cải thiện độ chính xác.
   - Tăng cường data augmentation hoặc thu thập thêm dữ liệu.

4. ỨNG DỤNG THỰC TẾ:
   - Có thể tích hợp vào trình duyệt như một extension cảnh báo website lừa đảo.
   - Triển khai trên server để quét hàng loạt URL.
""".format(
    acc * 100,
    recall_per_class[1] * 100 if len(recall_per_class) > 1 else 0,
    params_million,
    avg_inference_time_per_sample
))

# ============================================
# GIAI ĐOẠN 16: HÀM DỰ ĐOÁN ẢNH ĐƠN LẺ (DEMO)
# ============================================

def predict_single_image(image_path, model, class_names, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Dự đoán một ảnh đơn lẻ (có thể dùng để demo sau này)
    """
    from tensorflow.keras.preprocessing import image
    
    # Đọc và tiền xử lý ảnh
    img = image.load_img(image_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Chuẩn hóa
    
    # Dự đoán
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    result = {
        'class': class_names[predicted_class],
        'confidence': confidence,
        'is_phishing': predicted_class == 1,
        'probabilities': {
            class_names[0]: predictions[0][0],
            class_names[1]: predictions[0][1]
        }
    }
    return result

# Demo thử (nếu có ảnh trong thư mục test)
log_to_file_and_console("\n" + "=" * 60)
log_to_file_and_console("DEMO DỰ ĐOÁN TRÊN MỘT SỐ ẢNH MẪU")
log_to_file_and_console("=" * 60)

# Tìm một số ảnh mẫu để demo
demo_images = []
for class_name in class_names:
    class_dir = os.path.join(test_dir, class_name)
    if os.path.exists(class_dir):
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if files:
            demo_images.append(os.path.join(class_dir, files[0]))

for img_path in demo_images:
    if os.path.exists(img_path):
        result = predict_single_image(img_path, model, class_names)
        log_to_file_and_console(f"\nẢnh: {os.path.basename(img_path)}")
        log_to_file_and_console(f"  - Dự đoán: {result['class']}")
        log_to_file_and_console(f"  - Độ tin cậy: {result['confidence'] * 100:.2f}%")
        log_to_file_and_console(f"  - Là phishing: {'CÓ' if result['is_phishing'] else 'KHÔNG'}")

# ============================================
# KẾT THÚC: ĐÓNG FILE VÀ HIỂN THỊ
# ============================================

report_file.close()

print("\n" + "=" * 60)
print("HOÀN THÀNH! CÁC FILE KẾT QUẢ:")
print("=" * 60)
print(f"  - Báo cáo chi tiết: {REPORT_TXT_PATH}")
print(f"  - Mô hình đã lưu: {MODEL_SAVE_PATH}")
print(f"  - Đồ thị loss: {LOSS_PLOT_PATH}")
print(f"  - Đồ thị accuracy: {ACC_PLOT_PATH}")
print(f"  - Ma trận nhầm lẫn: {CM_PLOT_PATH}")
print("\nNHẤN PHÍM BẤT KỲ ĐỂ ĐÓNG CÁC ĐỒ THỊ...")

plt.show()