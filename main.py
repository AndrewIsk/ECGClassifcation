from fastapi import FastAPI , File, UploadFile
from typing import Annotated
import wfdb
import os
import tempfile
import torch
from torch import nn
from fastapi import HTTPException
#from ECG_Thing import model


class CNN(nn.Module):
  def __init__(self, in_channels, num_classes):
    super(CNN, self).__init__()

    self.features = nn.Sequential(
      nn.Conv1d(12, 32, kernel_size=7, padding=3),
      nn.ReLU(),
      nn.MaxPool1d(2),

      nn.Conv1d(32, 64, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.MaxPool1d(2),

      nn.Conv1d(64, 128, kernel_size=3, padding=1),
      nn.ReLU(),

      nn.AdaptiveAvgPool1d(1)
    )

    self.classifier = nn.Linear(128, num_classes)


  def forward(self, x):
    x = self.features(x)
    x = x.squeeze(-1)
    x = self.classifier(x)
    return x

app = FastAPI()

@app.get("/health/")
def health():
    return {"status": "OK"}

@app.post("/files/")
def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}

@app.post("/uploadfile/")
async def create_upload_file(datFile: UploadFile, heaFile: UploadFile):
  if not datFile.filename.endswith(".dat"):
    raise HTTPException(status_code=400, detail="datFile must be a .dat file")
  if not heaFile.filename.endswith(".hea"):
    raise HTTPException(status_code=400, detail="heaFile must be a .hea file")

  with tempfile.TemporaryDirectory() as tmpdir:
    # WFDB requires matching base filenames, e.g. "record100.dat" + "record100.hea"
    recordName = os.path.splitext(datFile.filename)[0]

    dat_path = os.path.join(tmpdir, datFile.filename)
    hea_path = os.path.join(tmpdir, heaFile.filename)
    with open(dat_path, "wb") as f:
        f.write(await datFile.read())
    with open(hea_path, "wb") as f:
        f.write(await heaFile.read())

    path = os.path.join(tmpdir, recordName)    
    onlySignal, _ = wfdb.rdsamp(path)
    state_dict = torch.load('best_model.pth')

    model = CNN(in_channels=12, num_classes=5)
    model.load_state_dict(state_dict)
    outputs = model(torch.rot90((torch.from_numpy(onlySignal).float()), k=1))
    percentages = (torch.softmax(outputs, dim=0)*100).tolist()
    labels = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    result = dict(zip(labels, percentages))
    return {"Outputs": result}