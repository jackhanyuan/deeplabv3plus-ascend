# deeplabv3plus ascend
Deeplabv3+ om model inference program on the Huawei Ascend platform

All programs passed the test on Huawei `Atlas 300I` inference card (`Ascend 310 AI CPU`, `CANN 5.0.2`, `npu-smi 21.0.2`).

You can run demo by `python detect_deeplabv3plus_ascend.py`.

## Export om model 
(1) Training your Deeplabv3+ model by [bubbliiiing/deeplabv3-plus-pytorch](https://github.com/bubbliiiing/deeplabv3-plus-pytorch). Then export the pytorch model to onnx format.
```bash
# in deeplabv3-plus-pytorch root path, exporting pth model to onnx model.
python export_onnx.py
```

(2) On the Huawei Ascend platform, using the `atc` tool convert the onnx model to om model.
```bash
# on Ascend 310 AI CPU, exporting onnx model to om model.
atc --input_shape="images:1,3,512,512" --input_format=NCHW --output="deeplab_mobilenetv2" --soc_version=Ascend310 --framework=5 --model="deeplab_mobilenetv2.onnx" --output_type=FP32 
```

## Inference by Ascend NPU
(1) Clone repo and move `*.om model` to `deeplabv3plus-ascend/ascend/*.om`.
```bash
git clone git@github.com:jackhanyuan/deeplabv3plus-ascend.git
mv deeplab_mobilenetv2.om deeplabv3plus-ascend/ascend/
```

(2) Edit label file in `deeplabv3plus-ascend/ascend/deeplabv3plus.label`.


(3) Run inference program.
```bash
python detect_deeplabv3plus_ascend.py
```
The result will save to `img_out` folder.
