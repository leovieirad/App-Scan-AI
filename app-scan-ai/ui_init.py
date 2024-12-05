import customtkinter as ctk
from PIL import Image


class AppScanAI:

    def __init__(self):
        
        self.main_window = ctk.CTk()
        self.main_window.geometry("1080x580")
        self.main_window.resizable(width = False, height = False)
        self.main_window.title("Scan AI")

    
        self.folder = None
        self.folder_logo = None
        self.progress_bar = None
        self.update_progress = None
        self.folder_without_logo = None
        self.model_folder = None
        self.models = []
        self.image_paths = []
        self.image_logo_paths = []
        self.image_files = []
        self.image_without_logo = []
        self.metadata_logos = []
        self.format_data = []
        self.divide_widgets = []
        self.feats = []
        self.keep = []
        self.svm_detections = {}
        
        self.image_logo = ctk.CTkImage(dark_image = Image.open(r"C:\Users\nubin\Desktop\Trabalho-app-scan-ai\Logo\Logo.jpg"), size = (200, 200))
        self.label_logo = ctk.CTkLabel(self.main_window, image = self.image_logo, text = "")
        self.label_logo.pack(pady = 80)

        self.button_enter = ctk.CTkButton(self.main_window, text = "Entrar", fg_color = "#8A2431", hover_color = "#D93D5F", width = 200, height =  50, font = ("Arial", 16), command = self.open_new_window)
        self.button_enter.pack(pady = 50)

        self.main_window.mainloop()
        
    
    def open_new_window(self):
        
        self.main_window.destroy()
        self.new_window = ctk.CTk()
        self.new_window.configure(fg_color = "#2F2F2F")
        self.new_window.geometry("1080x580")
        self.new_window.resizable(width = False, height = False)
        self.new_window.title("Scan AI")
        
        from ui_tabs import create_tabs
        create_tabs(self)

        self.new_window.mainloop()