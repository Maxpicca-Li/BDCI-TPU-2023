# 量化校准
run_calibration.py r-esrgan4x.mlir \
    --dataset ../dataset/test \
    --input_num 2 \
    -o r-esrgan4x_pt_cali_table

# f16 部署
model_deploy.py \
 --mlir r-esrgan4x.mlir \
 --quantize F16 \
 --chip bm1684x \
 --model resrgan4x.bmodel

# i8 部署
model_deploy.py \
    --mlir r-esrgan4x.mlir \
    --quantize INT8 \
    --calibration_table r-esrgan4x_pt_cali_table.1 \
    --chip bm1684x \
    --test_input yolov5s_pt_in_f32.npz \
    --test_reference yolov5s_pt_top_outputs.npz \
    --tolerance 0.85,0.45 \
    --model yolov5s_pt_1684x_int8_sym.bmodel

