https://universe.roboflow.com/roboflow-100/lettuce-pallets 

1510 images

you can downloading using

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

We'll use the `train` data.