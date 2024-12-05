import os
import main
import machine_learning
import threading
import customtkinter as ctk
from customtkinter import filedialog


def create_tabs(app):
    
    my_values = ["Identificador de Logo", "Treinamento de modelo machine learning"]
    app.my_seg_button = ctk.CTkSegmentedButton(app.new_window, selected_hover_color = "#8A2431", selected_color = "#8A2431", unselected_color = "#8A2431", fg_color = "#8A2431", font = ("Arial", 14), values = my_values, command = lambda values: tab_change(values, app))
    app.my_seg_button.pack(pady = 10)
    tab_Logo(app)
    
    
def tab_Logo(app):
    clean_window(app)
    
    label_format = ctk.CTkLabel(app.new_window, text = "Formato de imagem suportado: ", font = ("Arial", 21), fg_color = "#2F2F2F")
    label_format.place(x = 64, y = 60)
    
    label_format = ctk.CTkLabel(app.new_window, text = ".jpg", font = ("Arial", 19), fg_color = "#2F2F2F")
    label_format.place(x = 64, y = 95)
    
    label_image = ctk.CTkLabel(app.new_window, text = "Selecionar imagens para detecção", font = ("Arial", 21), fg_color = "#2F2F2F")
    label_image.place(x = 64, y = 160)
    
    label_image_logo = ctk.CTkLabel(app.new_window, text = "Selecionar logos para detectar", font = ("Arial", 21), fg_color = "#2F2F2F")
    label_image_logo.place(x = 500, y = 300)
    
    label_folder = ctk.CTkLabel(app.new_window, text = "Selecionar pasta para salvar", font = ("Arial", 21), fg_color = "#2F2F2F")
    label_folder.place(x = 64, y = 300)
    
    label_image_path = ctk.CTkLabel(app.new_window, text = "Nenhuma imagem selecionado", font = ("Arial", 14), fg_color = "#2F2F2F")
    label_image_path.place(x = 64, y = 255)
    
    label_image_path_logo = ctk.CTkLabel(app.new_window, text = "Nenhuma logo selecionado", font = ("Arial", 14), fg_color = "#2F2F2F")
    label_image_path_logo.place(x = 500, y = 395)
    
    label_folder_select = ctk.CTkLabel(app.new_window, text = "Nenhuma pasta selecionada", font = ("Arial", 14), fg_color = "#2F2F2F")
    label_folder_select.place(x = 64, y = 395)
    
    label_result_convert = ctk.CTkLabel(app.new_window, text = "", font = ("Arial", 14), fg_color = "#2F2F2F")
    label_result_convert.place(x = 380, y = 490)
    
    label_selected_option = ctk.CTkLabel(app.new_window, text = "", font = ("Arial", 14), fg_color = "#2F2F2F")
    label_selected_option.place(x = 480, y = 160)
    
    app.progress_bar = ctk.CTkProgressBar(app.new_window, width = 300, progress_color = "#8A2431")
    app.progress_bar.place(x = 380, y = 520)
    app.progress_bar.set(0)
    
    
    def selectImage():
        app.image_paths
        paths = filedialog.askopenfilenames(initialdir = r"/Desktop", title = "Selecione as imagens")
            
        if paths:
            valid_images = []
            supported_formats = ['.jpg']
            for path in paths:
                if os.path.isfile(path):
                    file_extension = os.path.splitext(path)[1].lower()
                    if file_extension in supported_formats:
                        valid_images.append(path)
                else:
                    label_image_path.configure(text = "Formato de imagem não suportado", font = ("Arial", 17), text_color = "red", fg_color = "#2F2F2F")
            if valid_images:
                app.image_paths = valid_images
                label_image_path.configure(text = f"{len(app.image_paths)} imagens selecionadas", font = ("Arial", 17), text_color = "green", fg_color = "#2F2F2F")                     
          
            else:
                label_image_path.configure(text = "Erro: Nenhum arquivo de imagem válido", font = ("Arial", 17), text_color = "red", fg_color = "#2F2F2F")  
  
  
    def selectImage_Logo():
        app.image_logo_paths
        paths = filedialog.askopenfilenames(initialdir = r"/Desktop", title = "Selecione as imagens")
            
        if paths:
            valid_images = []
            supported_formats = ['.jpg']
            for path in paths:
                if os.path.isfile(path):
                    file_extension = os.path.splitext(path)[1].lower()
                    if file_extension in supported_formats:
                        valid_images.append(path)
                else:
                    label_image_path_logo.configure(text = "Formato de imagem não suportado", font = ("Arial", 17), text_color = "red", fg_color = "#2F2F2F")
            if valid_images:
                app.image_logo_paths = valid_images
                label_image_path_logo.configure(text = f"{len(app.image_logo_paths)} imagens selecionadas", font = ("Arial", 17), text_color = "green", fg_color = "#2F2F2F")                   
          
            else:
                label_image_path_logo.configure(text = "Erro: Nenhum arquivo de imagem válido", font = ("Arial", 17), text_color = "red", fg_color = "#2F2F2F")  
    
       
    def selectFolder_Image():
        app.folder
        folder_path = filedialog.askdirectory(initialdir = r"/Desktop", title = "Selecione uma pasta")
        
        if os.path.isdir(folder_path):
            app.folder = folder_path
            label_folder_select.configure(text = f"Pasta selecionada \n {app.folder}", font = ("Arial", 17), text_color = "green", fg_color = "#2F2F2F")                  
        else:
            label_folder_select.configure(text = "Não é uma pasta válida", font = ("Arial", 17), text_color = "red", fg_color = "#2F2F2F")
    
    
    def machine_learning():
        if checkbox_var.get() == 1:      
            button_converter.place_forget()
        
            label_divide = ctk.CTkLabel(app.new_window, text = "Selecione o modelo", font = ("Arial", 21), fg_color = "#2F2F2F")
            label_divide.place(x = 500, y = 160)
            app.divide_widgets.append(label_divide)
            
            label_model = ctk.CTkLabel(app.new_window, text = "Nenhum modelo selecionado", font = ("Arial", 14), fg_color = "#2F2F2F")
            label_model.place(x = 500, y = 255)
            app.divide_widgets.append(label_model)
   
            
            def select_model():
                app.models
                models = filedialog.askopenfilenames(initialdir = r"C:\Users\nubin\Desktop\Modelo_machine_learning", title = "Selecione o modelo")
                if models:
                    valid_models = []
                    supported_formats = ['.joblib']
                    for model in models:
                        if os.path.isfile(model):
                            file_extension = os.path.splitext(model)[1].lower()
                            if file_extension in supported_formats:
                                valid_models.append(model)
                        else:
                            label_model.configure(text = "Formato de modelo não suportado", font = ("Arial", 17), text_color = "red", fg_color = "#2F2F2F")
                    if valid_models:
                        app.models = valid_models
                        label_model.configure(text = f"Modelos selecionados!", font = ("Arial", 17), text_color = "green", fg_color = "#2F2F2F")                  
                    else:
                        label_model.configure(text = "Não é um modelo válido", font = ("Arial", 17), text_color = "red", fg_color = "#2F2F2F")
            
                app.divide_widgets.append(label_model)
            
            
            def machine_learning():
            
                if app.image_paths and app.folder and app.image_logo_paths and app.models and app.progress_bar:
                    threading.Thread(target = run_machine_learning, args = (app,)).start()
                else:
                    label_result_convert.configure(text = "Erro: Um ou mais parâmetros estão faltando para a conversão.", font = ("Arial", 14), text_color = "red", fg_color = "#2F2F2F")
    
    
            def run_machine_learning(app):
                if app.image_paths and app.folder and app.image_logo_paths and app.models and app.progress_bar:                               
            
                    def update_progress(progress):
                        app.progress_bar.set(progress)
                        app.new_window.update_idletasks()
                         
                main.extract_data_logos(app.image_logo_paths, app.image_paths, app)
                main.detect_logo(app.image_paths, app.image_logo_paths, app.folder, app.models, update_progress, app)                    
                app.new_window.after(1, lambda: label_result_convert.configure(text = f"As logos foram identificadas e estão na pasta: {app.folder}", font = ("Arial", 14), text_color = "green", fg_color = "#2F2F2F"))
            
            
            button_model = ctk.CTkButton(app.new_window, hover_color = "#D93D5F", fg_color = "#8A2431", text = "Selecionar modelo", width = 100, height = 50, font = ("Arial", 16), command = select_model)
            button_model.place(x = 500, y = 200)
            app.divide_widgets.append(button_model)
            
            button_machine_learning = ctk.CTkButton(app.new_window, hover_color = "#D93D5F", fg_color = "#8A2431", text = "Identificar", width = 140, height = 50, font = ("Arial", 20), command = machine_learning)
            button_machine_learning.place(x = 455, y = 440)
            app.divide_widgets.append(button_machine_learning)
        
        else:
            button_converter.place(x = 455, y = 440)
            for widget in app.divide_widgets:
                widget.destroy()
            app.divide_widgets.clear() 
    
    
    def identify_logo():
        if app.image_paths and app.folder and app.image_logo_paths:
            threading.Thread(target = run_identify_logo, args = (app,)).start()
        else:
            label_result_convert.configure(text = "Erro: Um ou mais parâmetros estão faltando.", font = ("Arial", 14), text_color = "red", fg_color = "#2F2F2F")

    
    def run_identify_logo(app):
        if app.image_paths and app.folder and app.image_logo_paths:                               
            try:
                def update_progress(progress):
                    app.progress_bar.set(progress)
                    app.new_window.update_idletasks()
                    
                main.extract_data_logos(app.image_logo_paths, app.image_paths, app)
                main.detect_logo(app.image_paths, app.image_logo_paths, app.folder, update_progress, app)
            
                app.new_window.after(1, lambda: label_result_convert.configure(text = f"As logos foram identificadas e estão na pasta: {app.folder}", font = ("Arial", 14), text_color = "green", fg_color = "#2F2F2F"))
        
            except Exception as e:
                label_result_convert.configure(text = f"Erro durante a identificação das logos: {str(e)}", font = ("Arial", 14), text_color = "red", fg_color = "#2F2F2F")
        else:
            label_result_convert.configure(text = "Erro: Um ou mais parâmetros estão faltando.", font = ("Arial", 14), text_color = "red", fg_color = "#2F2F2F")


    checkbox_var = ctk.IntVar()
    checkbox = ctk.CTkCheckBox(app.new_window, text = "Machine Learning", fg_color = "#8A2431", hover_color = "#8A2431", variable = checkbox_var, font = ("Arial", 19), command = machine_learning)
    checkbox.place(x = 500, y = 130)

    button_image = ctk.CTkButton(app.new_window, hover_color = "#D93D5F", fg_color = "#8A2431", text = "Selecionar Imagem", width = 100, height = 50, font = ("Arial", 16), command = selectImage)
    button_image.place(x = 64, y = 200)
    
    button_image_logo = ctk.CTkButton(app.new_window, hover_color = "#D93D5F", fg_color = "#8A2431", text = "Selecionar Imagem", width = 100, height = 50, font = ("Arial", 16), command = selectImage_Logo)
    button_image_logo.place(x = 500, y = 340)
    
    button_folder = ctk.CTkButton(app.new_window, hover_color = "#D93D5F", fg_color = "#8A2431", text = "Selecionar Pasta", width = 100, height = 50, font = ("Arial", 16), command = selectFolder_Image)
    button_folder.place(x = 64, y = 340)
    
    button_converter = ctk.CTkButton(app.new_window, hover_color = "#D93D5F", fg_color = "#8A2431", text = "Identificar", width = 140, height = 50, font = ("Arial", 20), command = identify_logo)
    button_converter.place(x = 455, y = 440)
    

def tab_machine_learning(app):
    
    clean_window(app)
    
    label_format = ctk.CTkLabel(app.new_window, text = "Formato de imagem suportado: ", font = ("Arial", 21), fg_color = "#2F2F2F")
    label_format.place(x = 64, y = 60)
    
    label_format = ctk.CTkLabel(app.new_window, text = ".jpg", font = ("Arial", 19), fg_color = "#2F2F2F")
    label_format.place(x = 64, y = 95)
    
    label_data = ctk.CTkLabel(app.new_window, text = "Selecionar dados com a logo", font = ("Arial", 21), fg_color = "#2F2F2F")
    label_data.place(x = 64, y = 160)
    
    label_folder_logo = ctk.CTkLabel(app.new_window, text = "Nenhuma pasta selecionada", font = ("Arial", 14), fg_color = "#2F2F2F")
    label_folder_logo.place(x = 64, y = 255)
    
    label_folder_without_logo = ctk.CTkLabel(app.new_window, text = "Selecionar dados sem a logo", font = ("Arial", 21), fg_color = "#2F2F2F")
    label_folder_without_logo.place(x = 64, y = 300)
    
    label_without_logo = ctk.CTkLabel(app.new_window, text = "Nenhuma logo selecionado", font = ("Arial", 14), fg_color = "#2F2F2F")
    label_without_logo.place(x = 64, y = 400)

    label_folder = ctk.CTkLabel(app.new_window, text = "Selecionar pasta para salvar", font = ("Arial", 21), fg_color = "#2F2F2F")
    label_folder.place(x = 500, y = 300)
    
    label_save_folder = ctk.CTkLabel(app.new_window, text = "Nenhuma pasta selecionada", font = ("Arial", 14), fg_color = "#2F2F2F")
    label_save_folder.place(x = 500, y = 400)
              
    label_result_convert = ctk.CTkLabel(app.new_window, text = "", font = ("Arial", 14), fg_color = "#2F2F2F")
    label_result_convert.place(x = 380, y = 490)
    
    label_selected_option = ctk.CTkLabel(app.new_window, text = "", font = ("Arial", 14), fg_color = "#2F2F2F")
    label_selected_option.place(x = 480, y = 160)
    
    app.progress_bar = ctk.CTkProgressBar(app.new_window, width = 300, progress_color = "#8A2431")
    app.progress_bar.place(x = 380, y = 520)
    app.progress_bar.set(0)
    
    
    def select_folder_logo():
        
        folder_logo = filedialog.askdirectory(initialdir = r"/Desktop", title = "Selecione uma pasta")
        if os.path.isdir(folder_logo):
            app.folder_logo = folder_logo
            label_folder_logo.configure(text = f"Pasta selecionada \n {app.folder_logo}", font = ("Arial", 17), text_color = "green", fg_color = "#2F2F2F")                  
            image_files = [os.path.join(folder_logo, f) for f in os.listdir(folder_logo) if f.lower().endswith(('.jpg'))]
            if image_files:
                app.image_logo_paths = image_files
                label_folder_logo.configure(text = f"Arquivos de imagem encontrados!", font = ("Arial", 14), text_color = "green")
            else:
                label_folder_logo.configure(text = "Nenhum arquivo de imagem encontrado na pasta.", font = ("Arial", 14), text_color = "red")
        else:
            label_folder_logo.configure(text = "Não é uma pasta válida", font = ("Arial", 17), text_color = "red", fg_color = "#2F2F2F")
                    
  
    def select_folder_without_logo():
        
        folder_logo = filedialog.askdirectory(initialdir = r"/Desktop", title = "Selecione uma pasta")
        if os.path.isdir(folder_logo):
            app.folder_without_logo = folder_logo
            label_without_logo.configure(text = f"Pasta selecionada \n {app.folder_without_logo}", font = ("Arial", 17), text_color = "green", fg_color = "#2F2F2F")                  
            image_without_logo = [os.path.join(folder_logo, f) for f in os.listdir(folder_logo) if f.lower().endswith(('.jpg'))]
            if image_without_logo:
                app.image_without_logo = image_without_logo
                label_without_logo.configure(text = f"Arquivos de imagem encontrados!", font = ("Arial", 14), text_color = "green")
            else:
                label_without_logo.configure(text = "Nenhum arquivo de imagem encontrado na pasta.", font = ("Arial", 14), text_color = "red")
        else:
            label_without_logo.configure(text = "Não é uma pasta válida", font = ("Arial", 17), text_color = "red", fg_color = "#2F2F2F")
                    
       
    def select_model_folder():
        app.model_folder
        folder_path = filedialog.askdirectory(initialdir = r"C:\Users\nubin\Desktop\Modelo_machine_learning", title = "Selecione uma pasta")
        
        if os.path.isdir(folder_path):
            app.model_folder = folder_path
            label_save_folder.configure(text = f"Pasta selecionada \n {app.model_folder}", font = ("Arial", 17), text_color = "green", fg_color = "#2F2F2F")                  
        else:
            label_save_folder.configure(text = "Não é uma pasta válida", font = ("Arial", 17), text_color = "red", fg_color = "#2F2F2F")
    
    
    def generate_report():
        if checkbox_var.get() == 1:
            button_converter.place_forget()
            
            def identify_logo():
                if app.folder_logo and app.folder_without_logo and app.model_folder:
                    threading.Thread(target = run_identify_logo, args = (app,)).start()
                else:
                    label_result_convert.configure(text = "Erro: Um ou mais parâmetros estão faltando.", font = ("Arial", 14), text_color = "red", fg_color = "#2F2F2F")
                app.divide_widgets.append(label_result_convert)
            
             
            def run_identify_logo(app):
                if app.folder_logo and app.folder_without_logo and app.model_folder:                               
                    try:
                        def update_progress(progress):
                            app.progress_bar.set(progress)
                            app.new_window.update_idletasks()
                            
                        feats, vocabulary, grid_search, labels, metrics, training_time, inference_time = machine_learning.train_svm_kmeans(app.folder_logo, app.folder_without_logo, app.model_folder, update_progress, condicional = 1)
                        machine_learning.generate_svm_report(app.model_folder, feats, vocabulary, grid_search, labels, metrics, training_time, inference_time)
                        
                        app.new_window.after(1, lambda: label_result_convert.configure(text = f"Modelo e relátorio foram criados: {app.model_folder}", font = ("Arial", 14), text_color = "green", fg_color = "#2F2F2F"))
                
                    except Exception as e:
                        label_result_convert.configure(text = f"Erro durante a identificação das logos: {str(e)}", font = ("Arial", 14), text_color = "red", fg_color = "#2F2F2F")
                else:
                    label_result_convert.configure(text = "Erro: Um ou mais parâmetros estão faltando.", font = ("Arial", 14), text_color = "red", fg_color = "#2F2F2F")
                app.divide_widgets.append(label_result_convert)

            identify_and_generate_report_button = ctk.CTkButton(app.new_window, hover_color = "#D93D5F", fg_color = "#8A2431", text = "Treinar", width = 140, height = 50, font = ("Arial", 20), command = identify_logo)
            identify_and_generate_report_button.place(x = 455, y = 440)
            app.divide_widgets.append(identify_and_generate_report_button)
            
        else:
            button_converter.place(x = 455, y = 440)
            for widget in app.divide_widgets:
                widget.destroy()
            app.divide_widgets.clear() 


    def identify_logo():
        if app.folder_logo and app.folder_without_logo and app.model_folder:
            threading.Thread(target = run_identify_logo, args = (app,)).start()
        else:
            label_result_convert.configure(text = "Erro: Um ou mais parâmetros estão faltando.", font = ("Arial", 14), text_color = "red", fg_color = "#2F2F2F")

    
    def run_identify_logo(app):
        if app.folder_logo and app.folder_without_logo and app.model_folder:                               
            try:
                def update_progress(progress):
                    app.progress_bar.set(progress)
                    app.new_window.update_idletasks()
                    
                machine_learning.train_svm_kmeans(app.folder_logo, app.folder_without_logo, app.model_folder, update_progress, condicional = 2)
                
                app.new_window.after(1, lambda: label_result_convert.configure(text = f"Modelo treinado!: {app.model_folder}", font = ("Arial", 14), text_color = "green", fg_color = "#2F2F2F"))
        
            except Exception as e:
                label_result_convert.configure(text = f"Erro durante a identificação das logos: {str(e)}", font = ("Arial", 14), text_color = "red", fg_color = "#2F2F2F")
        else:
            label_result_convert.configure(text = "Erro: Um ou mais parâmetros estão faltando.", font = ("Arial", 14), text_color = "red", fg_color = "#2F2F2F")


    checkbox_var = ctk.IntVar()
    checkbox = ctk.CTkCheckBox(app.new_window, text = "Gerar Relátorio", fg_color = "#8A2431", hover_color = "#8A2431", variable = checkbox_var, font = ("Arial", 19), command = generate_report)
    checkbox.place(x = 500, y = 160)

    button_folder_logo = ctk.CTkButton(app.new_window, hover_color = "#D93D5F", fg_color = "#8A2431", text = "Selecionar Dados", width = 100, height = 50, font = ("Arial", 16), command = select_folder_logo)
    button_folder_logo.place(x = 64, y = 195)
    
    button_folder_without_logo = ctk.CTkButton(app.new_window, hover_color = "#D93D5F", fg_color = "#8A2431", text = "Selecionar Dados", width = 100, height = 50, font = ("Arial", 16), command = select_folder_without_logo)
    button_folder_without_logo.place(x = 64, y = 345)
    
    button_folder = ctk.CTkButton(app.new_window, hover_color = "#D93D5F", fg_color = "#8A2431", text = "Selecionar Pasta", width = 100, height = 50, font = ("Arial", 16), command = select_model_folder)
    button_folder.place(x = 500, y = 345)
    
    button_converter = ctk.CTkButton(app.new_window, hover_color = "#D93D5F", fg_color = "#8A2431", text = "Treinar", width = 140, height = 50, font = ("Arial", 20), command = identify_logo)
    button_converter.place(x = 455, y = 440)


def clean_window(app):

    for widget in app.new_window.winfo_children():
        if widget not in [app.my_seg_button]:
            widget.destroy()


def tab_change(values, app):
    clean_window(app)

    if values == "Identificador de Logo":
        tab_Logo(app)
    elif values == "Treinamento de modelo machine learning":
        tab_machine_learning(app)