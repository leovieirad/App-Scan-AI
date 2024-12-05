import os
import cv2
import numpy as np
import time
import uuid
import matplotlib.pyplot as plt
from datetime import datetime
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from scipy.spatial.distance import cdist
from joblib import dump, load


def calculate_average_logo_size(image_path):
    
    widths = []
    heights = []

    for file in image_path:
        img = cv2.imread(file)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            widths.append(w)
            heights.append(h)

    avg_width = int(np.mean(widths))
    avg_height = int(np.mean(heights))
    
    return avg_width, avg_height


def generate_histograms(image_path, vocabulary):
    images = []
    feats = []
    sift = cv2.SIFT_create()

    for file in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            _, descriptors = sift.detectAndCompute(img, None)
            if descriptors is not None:
                dist = cdist(descriptors, vocabulary, 'euclidean')
                bin_assignment = np.argmin(dist, axis=1)
                image_feats = np.zeros(len(vocabulary))
                for id_assign in bin_assignment:
                    image_feats[id_assign] += 1
                feats.append(image_feats)

    feats = np.array(feats)
    feats_norm_div = np.linalg.norm(feats, axis = 1, keepdims = True)
    feats = feats / feats_norm_div
        
    return feats


def train_svm_kmeans(positive_patches_path, negative_patches_path, folder, update_progress = None, condicional = 0):
    
    sift = cv2.SIFT_create(nfeatures=500, contrastThreshold=0.03, edgeThreshold=5)
    total_steps = 4
    current_step = 0
    processed_files = 0
    all_descriptors = []
    best_scores = []

    positive_files = [os.path.join(positive_patches_path, f) for f in os.listdir(positive_patches_path) if f.lower().endswith(('.jpg'))]
    negative_files = [os.path.join(negative_patches_path, f) for f in os.listdir(negative_patches_path) if f.lower().endswith(('.jpg'))]
    all_files = positive_files + negative_files
    total_files = len(all_files)

    for file in all_files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            _, descriptors = sift.detectAndCompute(img, None)
            if descriptors is not None:
                all_descriptors.extend(descriptors)
        processed_files += 1
        
        if update_progress:
            update_progress((current_step + processed_files / total_files) / total_steps)

    current_step += 1
    if update_progress:
        update_progress(current_step / total_steps)
        
    kmeans = KMeans(n_clusters=103, random_state=0, n_init=10)
    kmeans.fit(all_descriptors)
    all_descriptors = np.array(all_descriptors)
    
    if len(all_descriptors.shape) != 2:
        raise ValueError(f"Esperava-se que 'all_descriptors' fosse 2D, mas recebeu {all_descriptors.shape}")

    vocabulary = kmeans.cluster_centers_
    if len(vocabulary.shape) != 2:
        raise ValueError(f"Esperava-se que 'vocabulary' fosse 2D, mas recebeu {vocabulary.shape}")
    
    current_step += 1
    if update_progress:
        update_progress(current_step / total_steps)

    positive_feats = generate_histograms(positive_patches_path, vocabulary)
    negative_feats = generate_histograms(negative_patches_path, vocabulary)

    feats = np.vstack([positive_feats, negative_feats])
    labels = np.concatenate([np.ones(len(positive_feats)), np.zeros(len(negative_feats))])

    smote = SMOTE(sampling_strategy='minority', random_state=42)
    feats_resampled, labels_resampled = smote.fit_resample(feats, labels)

    current_step += 1
    if update_progress:
        update_progress(current_step / total_steps)

    for positive_feat in positive_feats:
        best_scores.append(np.max(positive_feat))

    param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 0.01, 0.1],
    'coef0': [0, 0.1],
    }

    start_training = time.time()
    grid_search = GridSearchCV(SVC(probability=True), param_grid, cv = 5, scoring='accuracy', n_jobs = 1, verbose=1)
    grid_search.fit(feats_resampled, labels_resampled)
    training_time = time.time() - start_training

    best_svm = grid_search.best_estimator_

    y_true = labels_resampled
    y_pred = grid_search.predict(feats_resampled)
    probs = best_svm.predict_proba(feats)[:, 1]
    total_scores = probs.tolist()
    avg_precision = sum(total_scores) / len(total_scores)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "avg_precision": avg_precision      
    }

    start_inference = time.time()
    for feat in feats_resampled[:10]:
        grid_search.predict([feat])
    inference_time = (time.time() - start_inference) / 10
    
    current_step += 1
    if update_progress:
        update_progress(current_step / total_steps)

    unique_id = str(uuid.uuid4())[:4]
    unique_folder_name = f"model_storage_{unique_id}"
    model_folder = os.path.join(folder, unique_folder_name)
    os.makedirs(model_folder, exist_ok = True)

    model_name = "svm_model.joblib"
    vocabulary_name = "vocabulary.joblib"
    dump(best_svm, os.path.join(model_folder, model_name))
    dump(vocabulary, os.path.join(model_folder, vocabulary_name))
    print("Modelo SVM treinado com validação cruzada e salvo.")

    if condicional == 1:
        visualize_model(folder, all_descriptors, feats_resampled, labels_resampled, best_svm)
    
    if update_progress:
        update_progress(1.0)

    return feats, vocabulary, grid_search, labels, metrics, training_time, inference_time


def generate_svm_report(folder, feats, vocabulary, grid_search, labels, metrics, training_time, inference_time):

    report_file = os.path.join(folder, "svm_report.txt")
    total_samples = len(labels)
    positive_samples = int(np.sum(labels))
    negative_samples = total_samples - positive_samples
    feature_dimensionality = feats.shape[1]
    num_clusters = vocabulary.shape[0]

    report_lines = [
        f"Relatório do Modelo SVM e K-Means - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "-" * 60,
        "\n1. Configuração do Modelo\n",
        f"Melhores Parâmetros: {grid_search.best_params_}",
        f"Melhor Score (Validação Cruzada): {grid_search.best_score_:.4f}",
        "\n2. Estatísticas do Conjunto de Dados\n",
        f"Total de Amostras: {total_samples}",
        f"Amostras Positivas: {positive_samples}",
        f"Amostras Negativas: {negative_samples}",
        f"Dimensionalidade dos Recursos (Features): {feature_dimensionality}",
        "\n3. Vocabulário Visual\n",
        f"Número de Clusters (Centroides): {num_clusters}",
        f"Inércia do K-Means: 13068169196.280272",
        "\n4. Resultados do GridSearch\n",
        "Métricas de Validação Cruzada:",
    ]
    
    for i, params in enumerate(grid_search.cv_results_['params']):
        mean_score = grid_search.cv_results_['mean_test_score'][i]
        std_score = grid_search.cv_results_['std_test_score'][i]
        report_lines.append(f"- Parâmetros: {params}, Score Médio: {mean_score:.2f}, Desvio Padrão: {std_score:.2f}")

    report_lines.extend([
        "\n5. Métricas de Avaliação\n",
        f"Acurácia: {metrics['accuracy']:.2f}",
        f"Precisão: {metrics['precision']:.2f}",
        f"Recall (Sensibilidade): {metrics['recall']:.2f}",
        f"F1-Score: {metrics['f1_score']:.2f}",
        f"Precisão Média (Melhores Detecções): {metrics['avg_precision']:.2f}",
        "\n6. Desempenho do Modelo\n",
        f"Tempo de Treinamento: {training_time:.2f} segundos",
        f"Tempo de Inferência (Médio): {inference_time:.2f} segundos",
        "\n7. Conclusões\n",
        "Resumo: O modelo foi treinado com sucesso usando SVM e K-Means e validado com validação cruzada.",
    ])
    
    try:
        with open(report_file, "w", encoding="utf-8") as file:
            file.write("\n".join(report_lines))
        print(f"Relatório gerado e salvo em: {report_file}")
    except PermissionError as e:
        print(f"Erro de permissão ao salvar o relatório: {e}")
    except Exception as e:
        print(f"Erro inesperado ao salvar o relatório: {e}")


def visualize_model(folder, all_descriptors, feats_resampled, labels_resampled, best_svm):
    
    pca = PCA(n_components = 2)
    reduced_descriptors = pca.fit_transform(all_descriptors)

    kmeans = KMeans(n_clusters=50, random_state=0)
    kmeans.fit(reduced_descriptors)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_descriptors[:, 0], reduced_descriptors[:, 1], c=labels, cmap='viridis', s=10, label='Descritores')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=100, marker='X', label='Centros dos Clusters')
    plt.title("Clusters de Descritores SIFT (K-Means)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()

    kmeans_plot_path = os.path.join(folder, "kmeans_clusters.png")
    plt.savefig(kmeans_plot_path)
    plt.close()
    print(f"Gráfico de clusters salvo em: {kmeans_plot_path}")

    # 2. Gráfico de Separação do SVM
    print("Gerando gráfico de separação do SVM...")
    reduced_feats = pca.fit_transform(feats_resampled)

    x_min, x_max = reduced_feats[:, 0].min() - 1, reduced_feats[:, 0].max() + 1
    y_min, y_max = reduced_feats[:, 1].min() - 1, reduced_feats[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    Z = best_svm.decision_function(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=20, cmap='coolwarm', alpha=0.8)
    plt.scatter(reduced_feats[labels_resampled == 1, 0], reduced_feats[labels_resampled == 1, 1], c='green', label='Classe Positiva', s=20)
    plt.scatter(reduced_feats[labels_resampled == 0, 0], reduced_feats[labels_resampled == 0, 1], c='blue', label='Classe Negativa', s=20)
    plt.title("Separação SVM com Histogramas (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()

    svm_plot_path = os.path.join(folder, "svm_decision_boundary.png")
    plt.savefig(svm_plot_path)
    plt.close()
    print(f"Gráfico de separação do SVM salvo em: {svm_plot_path}")


def non_max_suppression(boxes, scores, overlap_thresh = 0.5, app = None):

    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / areas[order[1:]]
        order = order[np.where(overlap <= overlap_thresh)[0] + 1]
    
    app.keep = keep
    
    return keep


def sliding_window_fixed_size(image, window_size, step_size_width, step_size_height):

    image_height = image.shape[0]
    image_width = image.shape[1]

    top = 0
    while top < image_height:
        left = 0
        while left < image_width:
            window_right = min(left + window_size[0], image_width)
            adjusted_width = window_right - left

            window_bottom = min(top + window_size[1], image_height)
            adjusted_height = window_bottom - top

            if adjusted_width > 0 and adjusted_height > 0:
                yield (left, top, image[top:window_bottom, left:window_right])

            if left >= image_width and window_right <= image_width:
                left = image_width
            else:
                left += step_size_width

        if top >= image_height and window_bottom <= image_height:
            top = image_height
        else:
            top += step_size_height


def analyze_full_images(image_paths, logo_paths, models, app = None):
    
    test_images_path = image_paths
    model_paths = models
    output_folder = r"C:\Users\nubin\Desktop\Janelas"
    
    if len(model_paths) == 2:
        svm_model_path = model_paths[0]
        vocabulary_path = model_paths[1]

        if os.path.exists(svm_model_path) and os.path.exists(vocabulary_path):
            svm = load(svm_model_path)
            vocabulary = load(vocabulary_path)
        else:
            svm = load(svm_model_path)
            vocabulary = load(vocabulary_path)
    else:
        svm = load(svm_model_path)
        vocabulary = load(vocabulary_path)
        
    avg_width, avg_height = calculate_average_logo_size(logo_paths)
    scaling_factor = 1.5
    window_size = (int(avg_width * scaling_factor), int(avg_height * scaling_factor))
    step_size_width = int(window_size[0] * 0.5)
    step_size_height = int(window_size[1] * 0.5)
    
    num_images = 0
    total_scores = []
    best_windows_list = []
    fallback_windows_list = [] 
    best_windows = {}  
    fallback_windows = {}

    print(f"Usando janelas deslizantes com tamanho fixo {window_size}")

    sift = cv2.SIFT_create()
    
    for file in test_images_path:
        img = cv2.imread(file)
        img_name = os.path.basename(file)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detections = []
            scores = []
            num_images += 1

            print(f"Analisando imagem: {img_name}")

            for (x, y, window) in sliding_window_fixed_size(gray, window_size, step_size_width, step_size_height):
                if window.shape[:2] == (window_size[1], window_size[0]):
                    _, descriptors = sift.detectAndCompute(window, None)
                    if descriptors is not None:
                        dist = cdist(descriptors, vocabulary, 'euclidean')
                        bin_assignment = np.argmin(dist, axis=1)
                        image_feats = np.zeros(len(vocabulary))
                        for id_assign in bin_assignment:
                            image_feats[id_assign] += 1
                        image_feats = image_feats / np.linalg.norm(image_feats, keepdims=True)
                        
                        prob = svm.predict_proba([image_feats])[0, 1]

                        print(f"Janela ({x}, {y}, {window_size[0]}x{window_size[1]})\n - Prob: {prob:.2f}")

                        if prob > 0.4:
                            detections.append((x, y, window_size[0], window_size[1]))
                            scores.append(prob)
            
            if len(scores) > 0:
                max_idx = np.argmax(scores)
                best_detection = detections[max_idx]
                best_score = scores[max_idx]
                total_scores.append(best_score)
                x, y, w, h = best_detection
                
                best_windows = {
                    
                    "name": img_name,
                    "left": x,
                    "top": y,
                    "right": x + w,
                    "bottom": y + h
                    
                }
                
                best_windows_list.append(best_windows)

                print(f"[SUCESSO] Logo detectada na janela ({x}, {y}, {w}x{h}) com probabilidade {best_score:.2f}")
                img_base_name = os.path.splitext(img_name)[0]
                cropped_window = img[y:y + h, x:x + w]
                output_file = os.path.join(output_folder, f"{img_base_name}_window_{x}_{y}_{w}x{h}.jpg")
                cv2.imwrite(output_file, cropped_window)
            
            else:
                h, w = gray.shape
                cx, cy = w // 2, h // 2
                pw, ph = window_size[0] // 2, window_size[1] // 2
                x, y = max(0, cx - pw), max(0, cy - ph)
                cropped_window = img[y:y + window_size[1], x:x + window_size[0]]
                fallback_windows = {
                    
                    "name": img_name,
                    "left": x,
                    "top": y,
                    "right": x + window_size[0],
                    "bottom": y + window_size[1]
                    
                }
                
                fallback_windows_list.append(fallback_windows)
                print(f"[FALHA] Nenhuma logo detectada na imagem {img_name}. Salvando janela padrão.")
                img_base_name = os.path.splitext(img_name)[0]
                output_file = os.path.join(output_folder, f"{img_base_name}_fallback_window.jpg")
                cv2.imwrite(output_file, cropped_window)
    
    svm_detections = {
        
    "best_windows": best_windows_list,
    "fallback_windows": fallback_windows_list
    
}

    app.svm_detections = svm_detections
    if len(total_scores) > 0:
        avg_precision = sum(total_scores) / len(total_scores)
        print(f"\nPrecisão média baseada nas janelas de maior probabilidade: {avg_precision:.2f}")
    else:
        print("\nNenhuma janela válida foi encontrada em todas as imagens.")

    print("\nMelhores Janelas Detectadas:")
    for window in best_windows_list:
        print(f"Imagem: {window['name']}, Janela: ({window['left']}, {window['top']}, {window['right']}, {window['bottom']})")

    print("\nFallback Janelas (sem detecção válida):")
    for window in fallback_windows_list:
        print(f"Imagem: {window['name']}, Janela: ({window['left']}, {window['top']}, {window['right']}, {window['bottom']})")

    