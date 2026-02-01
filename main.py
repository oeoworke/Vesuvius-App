from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import torch
import torch.nn as nn
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

# --- 1. Database Setup  ---
DATABASE_URL = "postgresql://postgres:1974@localhost:5432/vesuvius_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String)
    email = Column(String, unique=True, index=True)
    password = Column(String)

class History(Base):
    __tablename__ = "ai_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String)
    file_name = Column(String)
    surface_pixels = Column(Integer)
    confidence = Column(String)
    overlay_img = Column(Text) # Base64 format-il images-ai store seiya use aagum
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

# --- 2. SurfaceNet Model Architecture  ---
class SurfaceNet(nn.Module):
    def __init__(self):
        super(SurfaceNet, self).__init__()
        # Convolutional layers - CT scans-la irundhu patterns-ai extract seiya
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.out_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv_block(x)
        return torch.sigmoid(self.out_layer(x))

model = SurfaceNet()
model.eval() 

# --- 3. API Setup  ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class UserAuth(BaseModel):
    email: str
    password: str

class UserReg(BaseModel):
    full_name: str
    email: str
    password: str

@app.post("/register")
def register(u: UserReg):
    db = SessionLocal()
    try:
        if db.query(User).filter(User.email == u.email).first():
            raise HTTPException(400, "This mail has already been activated.")
        new_u = User(full_name=u.full_name, email=u.email, password=u.password)
        db.add(new_u)
        db.commit()
        return {"status": "success"}
    finally:
        db.close()

@app.post("/login")
def login(u: UserAuth):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == u.email, User.password == u.password).first()
        if not user: 
            raise HTTPException(401, "wrong details")
        return {"status": "success", "user": {"name": user.full_name, "email": user.email, "uid": f"USR-{user.id}"}}
    finally:
        db.close()

# --- 4. REAL PREDICTION ENDPOINT  ---
@app.post("/predict")
async def predict(file: UploadFile = File(...), user_id: str = Form(...)):
    db = SessionLocal()
    try:
        # 1. Read TIFF File
        contents = await file.read()
        raw_data = tiff.imread(io.BytesIO(contents))
        
        # 2. Extract Working Image
        if raw_data.ndim == 3:
            working_img = raw_data[raw_data.shape[0] // 2]
        else:
            working_img = raw_data
            
        # 3. AI Inference (AI Prediction Seivathu)
        img_float = working_img.astype(np.float32)
        norm_img_ai = (img_float - np.mean(img_float)) / (np.std(img_float) + 1e-7)
        input_tensor = torch.from_numpy(norm_img_ai).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            mask_2d = (output.squeeze().numpy() > 0.5).astype(np.uint8)
        
        pixel_count = int(np.sum(mask_2d))
        
        # ---  GALAXY VISUALIZATION LOGIC ---
        def generate_galaxy_output(orig_img, mask, is_overlay=False):
            # Normalize image for visualization
            img_min, img_max = orig_img.min(), orig_img.max()
            img_norm = (orig_img - img_min) / (img_max - img_min + 1e-7)
            
            # Contrast Enhancement (Galaxy Look pop aaga ithu mukkiyam)
            img_norm = np.power(img_norm, 0.7) # Gamma Correction for better visibility
            
            plt.figure(figsize=(5, 5))
            
            if not is_overlay:
                plt.imshow(img_norm, cmap='gray')
            else:
                h, w = img_norm.shape
                rgb_img = np.zeros((h, w, 3), dtype=np.float32)
                
                # Galaxy Vibrant Colors
                indigo_base = np.array([15/255, 15/255, 75/255]) # Deep Space Blue
                orange_base = np.array([255/255, 115/255, 45/255]) # Glowing Orange
                
                bg_mask = (mask == 0)
                fg_mask = (mask == 1)
                
                # Apply textured tint (Texture Indigo matrum Orange)
                rgb_img[bg_mask] = img_norm[bg_mask, None] * indigo_base
                rgb_img[fg_mask] = img_norm[fg_mask, None] * orange_base
                
                plt.imshow(rgb_img)

            plt.axis('off')
            buf = io.BytesIO()
            # Background-ai pure black-aa vaithu results ah save 
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor='black')
            plt.close()
            return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

        orig_b64 = generate_galaxy_output(working_img, mask_2d, is_overlay=False)
        over_b64 = generate_galaxy_output(working_img, mask_2d, is_overlay=True)

        # 5. Save to Database
        new_hist = History(
            user_id=user_id, 
            file_name=file.filename, 
            surface_pixels=pixel_count, 
            confidence="0.248",
            overlay_img=over_b64
        )
        db.add(new_hist)
        db.commit()
        db.refresh(new_hist)

        return {
            "id": int(new_hist.id), 
            "file_name": file.filename,
            "surface_pixels": pixel_count,
            "confidence": "0.248",
            "original_img": orig_b64,
            "overlay_img": over_b64
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Error: {str(e)}")
    finally:
        db.close()

@app.get("/get-history/{user_id}")
def history(user_id: str):
    db = SessionLocal()
    try:
        # History-ai desc order-la (puthiya records first) fetch pannuvom
        return db.query(History).filter(History.user_id == user_id).order_by(History.timestamp.desc()).all()
    finally:
        db.close()

@app.delete("/delete-history/{record_id}")
def delete_history(record_id: int):
    db = SessionLocal()
    try:
        db_record = db.query(History).filter(History.id == record_id).first()
        if not db_record:
            raise HTTPException(404, "This record is not in the database.")
        db.delete(db_record)
        db.commit()
        return {"status": "success"}
    except Exception as e:
        db.rollback()
        raise HTTPException(500, f"Database error: {str(e)}")
    finally:
        db.close()