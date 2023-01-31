
## Run benchmark

```
pip install -r requirements.txt
```

You can download some data using

```bash
mkdir data; cd data; curl -L "https://universe.roboflow.com/ds/MNwmAeNoBj?key=Hm9seGdI9X" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```

The folder should look something like

```
data
├── data.yaml
├── README.dataset.txt
├── README.roboflow.txt
├── test
│   ├── images
│   └── labels
├── train
│   ├── images
│   └── labels
└── valid
    ├── images
    └── labels
```


run

```
python main.py --help
```