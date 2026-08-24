# Kế hoạch làm Lab K4 Day 22 — DPO/ORPO Alignment

> Mục tiêu: hiểu được pipeline `SFT → preference data → DPO → evaluation → deploy`, hoàn thành **100/100 điểm core**, nhắm **+20/20 điểm bonus có chấm**, đồng thời giữ bài sạch, tái lập được và đủ chiều sâu để giải thích khi demo.

## 1. Nguồn chính thức và nguyên tắc làm bài

- Repository: <https://github.com/VinUni-AI20k/K4-Track3-Day22-DPO-ORPO-Alignment>
- Rubric: <https://github.com/VinUni-AI20k/K4-Track3-Day22-DPO-ORPO-Alignment/blob/main/rubric.md>
- Hardware guide: <https://github.com/VinUni-AI20k/K4-Track3-Day22-DPO-ORPO-Alignment/blob/main/HARDWARE-GUIDE.md>
- Reflection template: <https://github.com/VinUni-AI20k/K4-Track3-Day22-DPO-ORPO-Alignment/blob/main/submission/REFLECTION.md>
- Screenshot checklist: <https://github.com/VinUni-AI20k/K4-Track3-Day22-DPO-ORPO-Alignment/blob/main/submission/screenshots/README.md>
- Bonus challenge: <https://github.com/VinUni-AI20k/K4-Track3-Day22-DPO-ORPO-Alignment/blob/main/BONUS-CHALLENGE.md>
- SFT dataset đã chốt: <https://huggingface.co/datasets/5CD-AI/Vietnamese-alpaca-gpt4-gg-translated>

Nguyên tắc:

1. Chạy đúng pipeline gốc trước khi thêm cải tiến nghiên cứu.
2. Sau mỗi notebook, kiểm tra artifact và chụp bằng chứng ngay.
3. Không chỉ nhìn DPO loss; phải đọc riêng `chosen reward`, `rejected reward` và reward gap.
4. So sánh SFT và SFT+DPO trên cùng prompt, decoding config và judge rubric.
5. Không ghi kết quả “đẹp” theo cảm tính; giữ số liệu thực và giải thích failure mode nếu có.
6. Không commit API key, base-model weights, Hugging Face cache hoặc checkpoint thừa.
7. Repo nộp phải public cho đến khi có điểm.

## 2. Chiến lược điểm số

### Core 100/100

| Nhóm | Bằng chứng cần có | Điểm |
|---|---|---:|
| NB1 — SFT mini | Adapter đúng `r=16`, `lora_alpha=32`; loss giảm; có ít nhất một sample generation | 17 |
| NB2 — Preference data | Có `train.parquet` đúng schema; in ít nhất ba cặp hợp lệ | 12 |
| NB3 — DPO | Adapter riêng; reward gap tăng; vẽ và giải thích riêng chosen/rejected | 28 |
| NB4 — Eval | Ít nhất tám prompt × hai model; có win/loss/tie; gồm bốn helpfulness và bốn safety | 15 |
| Reproducibility | Setup sạch + `make pipeline` hoặc Colab Run-all chạy được | 5 |
| Reflection | Hoàn chỉnh, có số liệu thật, đủ độ dài và phân tích | 20 |
| Verify | `make verify` exit code 0 | 3 |
| **Tổng** | | **100** |

### Bonus có chấm: đường ngắn nhất đến +20/20

Ưu tiên bộ ba sau vì vừa đúng 20 điểm, vừa không phụ thuộc API trả phí:

| Bonus | Điểm | Artifact |
|---|---:|---|
| NB5 — Merge + GGUF deploy | +6 | GGUF Q4_K_M dưới 5 GB và llama.cpp smoke bằng tiếng Việt |
| NB6 — Benchmark | +8 | JSON kết quả + biểu đồ bốn benchmark + phân tích alignment tax |
| β-sweep | +6 | β ∈ {0.05, 0.1, 0.5}, biểu đồ reward gap/win-rate và ≥100 từ phân tích |
| **Tổng** | **+20** | Đạt trần bonus |

Các bonus dự phòng nếu một mục trên thất bại:

- Hugging Face Hub adapter + model card: +5.
- Cross-judge OpenAI/Anthropic: +4.
- GGUF Q4_K_M + Q5_K_M được publish: +3.
- MMLU full coverage: +3.
- Public W&B run: +2.

Bonus sáng tạo trong `BONUS-CHALLENGE.md` **không cộng điểm**, chỉ nên làm sau khi core và +20 rigor đã an toàn.

## 3. Chọn compute

### Khuyến nghị chính

- Nếu Kaggle/Colab cung cấp T4 16 GB: dùng tier `T4`, model `unsloth/Qwen2.5-3B-bnb-4bit`.
- Kaggle T4×2: dùng một GPU theo notebook T4; không giả định notebook tự tận dụng cả hai GPU.
- Nếu có L4/A100 ≥24 GB: dùng `BIGGPU`, model 7B.
- Không dùng GPU 4 GB local cho pipeline chấm chính; dùng local để đọc code, kiểm tra Markdown và quản lý Git.

Tier không quyết định điểm. Rubric chấm độ rõ ràng của before/after và bằng chứng, không chấm tốc độ hay model lớn tuyệt đối.

### Cấu hình mặc định cần giữ để dễ đối chiếu rubric

| Tham số | T4 | BigGPU |
|---|---:|---:|
| Base model | Qwen2.5-3B 4-bit | Qwen2.5-7B 4-bit |
| Max length | 512 | 1024 |
| Per-device batch | 1 | 1 |
| Gradient accumulation | 8 | 4 |
| LoRA rank | 16 | 16 |
| LoRA alpha | 32 | 32 |
| DPO beta | 0.1 | 0.1 |
| DPO learning rate | 5e-7 | 5e-7 |
| Epoch | 1 | 1 |

## 4. Pipeline tổng thể

```text
Fork/public repo
    ↓
Setup + GPU smoke test
    ↓
NB1: build SFT-mini checkpoint
    ↓
NB2: prepare preference pairs
    ↓
NB3: train DPO + diagnose reward curves
    ↓
NB4: compare SFT vs SFT+DPO + judge
    ↓
Core verify + Reflection
    ↓
NB5: merge + GGUF smoke
    ↓
NB6: benchmark + alignment-tax analysis
    ↓
β-sweep
    ↓
Final verify + screenshots + public submission
```

## 5. Phase 0 — Chuẩn bị repo và bằng chứng môi trường

### Việc làm

1. Fork/copy repository sang GitHub cá nhân.
2. Đặt repo public ngay từ đầu.
3. Clone repo cá nhân để mọi thay đổi push đúng chỗ.
4. Chọn một pipeline chính: T4 hoặc BigGPU. Không đổi tier giữa chừng nếu không ghi rõ.
5. Tạo `.env` từ `.env.example`; chỉ lưu biến không nhạy cảm vào repo.
6. API key phải nằm trong Kaggle Secrets, Colab Secrets hoặc environment variable; không ghi vào notebook output.

### Lệnh kiểm tra

```bash
bash setup-colab.sh
make smoke
```

Hoặc laptop Linux/WSL có NVIDIA ≥12 GB:

```bash
bash setup-laptop.sh
make smoke
```

### Phải hiểu

- SFT chỉ tối đa hóa likelihood của demonstration tốt.
- DPO cần cả `chosen` và `rejected`, so sánh policy với reference.
- Trong pipeline PEFT, TRL có thể tắt adapter để lấy reference forward pass trên cùng base model; không nhất thiết giữ hai bản base weights độc lập.
- DPO vẫn tốn activation memory cao hơn SFT vì chấm chosen/rejected và policy/reference.

### Artifact/checkpoint

- Chụp `01-setup-gpu.png` có tên GPU, CUDA và VRAM.
- Ghi GPU, tier, base model, dataset slice và chi phí vào Reflection §1 ngay.

## 6. Phase 1 — NB1: SFT-mini

### Mục tiêu học

Tạo policy khởi đầu biết instruction-following tiếng Việt trước khi học preference. DPO không thay thế hoàn toàn bước xây policy khởi đầu phù hợp.

### Dataset đã chốt cho SFT

Dùng `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` cho **NB1/SFT**. Dataset có một split `train`, 52.002 dòng và sáu cột song ngữ:

```text
input_en, input_vi, instruction_vi, output_vi, output_en, instruction_en
```

Ba cột cần dùng là:

| Cột nguồn | Cột chuẩn của notebook | Vai trò |
|---|---|---|
| `instruction_vi` | `instruction` | Yêu cầu của người dùng |
| `input_vi` | `input` | Ngữ cảnh bổ sung; có thể rỗng |
| `output_vi` | `output` | Câu trả lời demonstration |

Dataset này là **instruction/SFT dataset**, không phải preference dataset. Vì không có `chosen` và `rejected`, không dùng nó để thay dataset của NB2/NB3. Với DPO, tiếp tục dùng preference data mà repository chỉ định (Argilla UltraFeedback cleaned) để không phá schema và rubric.

### Sửa `notebooks/01_sft_mini.py`

Đổi cấu hình mặc định:

```python
SFT_DATASET = os.environ.get(
    "SFT_DATASET",
    "5CD-AI/Vietnamese-alpaca-gpt4-gg-translated",
)
SFT_SLICE = int(os.environ.get("SFT_SLICE", "1000"))
SEED = int(os.environ.get("SEED", "42"))
```

Không dùng trực tiếp formatter cũ trên dataset mới vì formatter cũ chờ `instruction`, `input`, `output`. Sau `load_dataset`, lọc mẫu lỗi, shuffle tái lập được rồi normalize tên cột:

```python
from datasets import load_dataset

raw_ds = load_dataset(SFT_DATASET, split="train")

raw_ds = raw_ds.filter(
    lambda x: bool((x.get("instruction_vi") or "").strip())
    and bool((x.get("output_vi") or "").strip())
)
raw_ds = raw_ds.shuffle(seed=SEED)
raw_ds = raw_ds.select(range(min(SFT_SLICE, len(raw_ds))))

def normalize_vietnamese_alpaca(row):
    return {
        "instruction": row["instruction_vi"].strip(),
        "input": (row.get("input_vi") or "").strip(),
        "output": row["output_vi"].strip(),
    }

ds = raw_ds.map(
    normalize_vietnamese_alpaca,
    remove_columns=raw_ds.column_names,
)
```

Sau bước này mới gọi formatter/chat template hiện có của notebook. Giữ `SFT_SLICE=1000` cho lần chạy chính để kết quả tương thích mục tiêu mini-lab và tiết kiệm thời gian; chỉ tăng lên 2.000–5.000 sau khi pipeline chuẩn đã chạy thành công.

### Quality audit trước khi train

Dataset là bản dịch tự động, vì vậy cần kiểm tra chất lượng thay vì giả định tất cả mẫu đều tốt. Thêm một cell in:

- Tên dataset, số mẫu trước/sau lọc và seed.
- Ba mẫu hoàn chỉnh gồm instruction, input và output.
- Tỷ lệ `input_vi` rỗng.
- Thống kê độ dài ký tự hoặc token: min, median, p95, max.
- Tỷ lệ instruction/output trùng lặp nếu có thời gian.

Đọc thủ công ít nhất 20 mẫu. Ghi lại các lỗi như câu dịch gượng, sai nghĩa, còn tiếng Anh, output không làm theo instruction hoặc mẫu quá dài. Nếu lọc thêm, viết điều kiện rõ ràng và giữ cùng seed để người chấm tái lập được. Không tự động xóa mọi mẫu có từ tiếng Anh vì code, thuật ngữ kỹ thuật và tên riêng có thể hợp lệ.

Lưu ý cho báo cáo/model card: trang dataset hiện không công bố dataset card hoặc license rõ ràng. Ghi `license: not specified on the dataset page at time of use`, không tự đoán license; nêu đây là giới hạn về provenance và khả năng phát hành lại model.

### Việc làm

```bash
make sft
```

Nếu chạy notebook tương tác, mở `notebooks/01_sft_mini.ipynb` và chạy từ đầu đến cuối theo thứ tự.

Trước khi train, xác nhận log hiển thị đúng:

```text
SFT dataset: 5CD-AI/Vietnamese-alpaca-gpt4-gg-translated
selected examples: 1000
columns after normalization: instruction, input, output
```

### Kiểm tra bắt buộc

- `adapters/sft-mini/adapter_config.json` tồn tại.
- Trong adapter config có `r: 16`, `lora_alpha: 32`.
- Loss giảm trong một epoch. Không cần từng step giảm tuyệt đối; cần xu hướng giảm rõ và giải thích noise mini-batch nếu có.
- In ít nhất một sample generation của SFT model.
- Sample generation phải dùng prompt tiếng Việt chưa xuất hiện trong ba mẫu quality-audit; lưu cả prompt, decoding config và output.
- Notebook `.ipynb` giữ nguyên output cells.

### Bằng chứng

- `02-sft-loss.png` được crop rõ: tiêu đề, trục step, loss, đường xu hướng.
- Trong Reflection §1 ghi dataset ID, `SFT_SLICE=1000`, seed, cách normalize ba cột và các rule lọc.
- Trong Reflection §2 ghi final loss, thời gian train và ít nhất một nhận xét về chất lượng bản dịch của dataset.

### Câu hỏi phải tự trả lời được

1. Vì sao DPO không nên bắt đầu trực tiếp từ base model thô trong lab này?
2. LoRA `r=16`, `alpha=32` thay đổi bao nhiêu tham số so với full fine-tuning?
3. Vì sao loss giảm không tự động chứng minh output hữu ích hơn?

## 7. Phase 2 — NB2: Preference data

### Mục tiêu học

Chuyển dữ liệu thành schema mà DPO thật sự sử dụng:

```text
prompt + chosen response + rejected response
```

### Việc làm

```bash
make data
```

### Kiểm tra bắt buộc

- `data/pref/train.parquet` tồn tại.
- Có đúng ba cột chính: `prompt`, `chosen`, `rejected`.
- Không có null/empty string ở ba cột.
- `chosen != rejected` cho mọi record.
- In ba ví dụ đầy đủ để kiểm tra thủ công.
- Ghi dataset name, số pair và filtering logic vào Reflection.

### Bonus chất lượng nên thêm mà không phá pipeline

- In thống kê số token chosen/rejected.
- Báo tỷ lệ chosen dài hơn rejected.
- In ba sample có độ dài lệch lớn để nhận diện length bias.
- Ghi seed và slice size chính xác.
- Nếu tự thêm dữ liệu tiếng Việt, lưu riêng; không trộn âm thầm vào baseline UltraFeedback.

### Câu hỏi phải tự trả lời được

1. Bradley–Terry model biến preference pair thành xác suất như thế nào?
2. Nếu chosen thường dài hơn rejected, model có thể học bias gì?
3. Preference label nhiễu ảnh hưởng DPO mạnh hơn SFT ra sao?

## 8. Phase 3 — NB3: DPO training

### Mục tiêu học

DPO tối ưu preference mà không cần huấn luyện reward model riêng.

Công thức cần hiểu:

```text
Δpolicy = log πθ(chosen|x) − log πθ(rejected|x)
Δref    = log πref(chosen|x) − log πref(rejected|x)
logit   = β(Δpolicy − Δref)
L_DPO   = −log σ(logit)
```

### Ý nghĩa của β

- β nhỏ: preference update aggressive hơn, reward gap có thể tăng nhanh nhưng dễ drift/overfit.
- β lớn: policy bị giữ gần reference hơn; an toàn hơn nhưng preference signal có thể yếu.
- Không chọn β chỉ dựa vào training loss; phải xem reward curves và evaluation.

### Việc làm

```bash
make dpo
```

### Cấu hình baseline phải ghi lại

```text
beta=0.1
learning_rate=5e-7
epochs=1
loss_type=sigmoid
LoRA r=16, alpha=32
seed=42
```

### Kiểm tra bắt buộc

- `adapters/dpo/adapter_config.json` tồn tại và là artifact riêng với SFT adapter.
- `adapters/dpo/dpo_metrics.json` tồn tại.
- Reward gap cuối train dương hoặc failure được giải thích trung thực.
- Plot chứa ba thông tin: chosen reward, rejected reward và reward gap.
- Không chỉ nói “gap tăng nên model tốt”.

### Cách đọc reward curves để lấy trọn điểm

| Hiện tượng | Diễn giải |
|---|---|
| Chosen tăng, rejected đứng yên/giảm | Tín hiệu alignment khá tốt |
| Chosen tăng nhẹ, rejected giảm mạnh | Preference separation tăng; cần kiểm tra output |
| Chosen giảm, rejected giảm nhanh hơn | Likelihood displacement; gap tăng nhưng chosen absolute likelihood giảm |
| Gap âm | Có thể data bị đảo, β/LR không phù hợp hoặc training chưa học |
| Hai curve phẳng | Có thể LR quá thấp, run quá ngắn hoặc adapter không thực sự train |

Reflection §3 phải trả lời bằng số cụ thể:

- Chosen reward đầu/cuối.
- Rejected reward đầu/cuối.
- Reward gap đầu/cuối.
- Shape theo step.
- Có likelihood displacement không?
- Điều đó có phù hợp với NB4 output không?

### Bằng chứng

- `03-dpo-reward-curves.png` có đủ legend và hai trục.
- Chụp rõ cả chosen/rejected và gap; đây là bằng chứng quan trọng nhất của lab.

### Câu hỏi phải tự trả lời được

1. DPO khác SFT ở objective nào?
2. Reference model ngăn policy drift như thế nào?
3. Reward gap tăng nhưng chosen reward giảm có phải thành công không?
4. Vì sao cần NB4 thay vì kết luận trực tiếp từ training curve?

## 9. Phase 4 — NB4: SFT vs SFT+DPO evaluation

### Mục tiêu học

Kiểm tra behavior change thực tế, không chỉ optimizer metrics.

### Việc làm

```bash
make eval
```

### Thiết kế so sánh công bằng

- Dùng đúng tám fixed prompts: bốn helpfulness và bốn safety.
- Cùng tokenizer, generation length và deterministic decoding.
- SFT-only và SFT+DPO phải nhận cùng prompt.
- Không sửa prompt sau khi đã nhìn output để làm DPO trông tốt hơn.
- Judge rubric gồm helpfulness, truthfulness, refusal appropriateness và length appropriateness.

### Nếu không có API key

Manual rubric vẫn được đủ điểm. Điền verdict thật thay vì để placeholder `MANUAL — fill in`.

### Nếu dùng API judge

- Temperature bằng 0.
- Lưu verbatim verdict cho ít nhất ba prompt.
- Không để lộ key trong output/screenshot.
- Đọc lại verdict thủ công, nhất là prompt khủng hoảng/tự hại.

### Kiểm tra bắt buộc

- `data/eval/side_by_side.jsonl` tồn tại.
- `data/eval/judge_results.json` tồn tại.
- Bảng có ít nhất 8 prompt × 2 responses.
- Có overall và tách riêng helpfulness/safety win-loss-tie.
- Mọi manual placeholder đã được thay bằng verdict thật.

### Bằng chứng

- `04-side-by-side-table.png`.
- `05-judge-output.png` hoặc `05-manual-rubric.png`.
- Nếu bảng dài, dùng `04a-...` và `04b-...`.

### Câu hỏi phải tự trả lời được

1. DPO cải thiện helpfulness hay safety rõ hơn?
2. Có over-refusal trên prompt lành tính không?
3. Có output dài hơn nhưng không thêm thông tin không?
4. Judge có thiên vị verbosity hoặc tone không?

## 10. Core checkpoint — khóa 100 điểm trước khi làm bonus

Chạy:

```bash
make verify
```

Nếu chưa pass, không chạy bonus.

Checklist core:

- [ ] NB1–NB4 đều có executed `.ipynb` và output cells.
- [ ] SFT adapter config tồn tại, đúng r/alpha.
- [ ] DPO adapter và metrics tồn tại.
- [ ] Preference parquet tồn tại.
- [ ] Side-by-side và judge results tồn tại.
- [ ] Reflection không còn placeholder.
- [ ] Có tối thiểu ba screenshot để `verify.py` pass.
- [ ] Repo tái lập được từ setup sạch.

Để nhắm max điểm, không dừng ở minimum của `verify.py`; làm theo rubric và screenshot guide chặt hơn.

## 11. Reflection — phần dễ mất 20 điểm

### Quy tắc an toàn

Rubric nói §3 và §6 phải ≥150 từ, trong khi template của §3 ghi ≥100 từ. Hãy theo yêu cầu chặt hơn:

- Reflection §3: ít nhất 150 từ.
- Reflection §6: ít nhất 150 từ.
- Bonus benchmark §7: ít nhất 150 từ.

### Nội dung từng phần

1. **Setup:** GPU, CUDA, model, dataset slice, tier, cost.
2. **DPO results:** thời gian, VRAM peak, final loss, reward gap, mean output length.
3. **Reward curves:** phân tích riêng chosen/rejected, likelihood displacement và curve shape.
4. **Qualitative comparison:** đủ tám row và win/loss/tie.
5. **β trade-off:** bảng ba β và hypothesis trước khi xem kết quả.
6. **Personal reflection:** chọn đúng một quyết định, nêu alternative, lý do, result và điều sẽ đổi.
7. **Benchmark:** bốn benchmark, delta, alignment tax, catastrophic forgetting và điểm bất ngờ.

### Cách viết có chiều sâu

Mỗi nhận định nên theo cấu trúc:

```text
Quan sát bằng số → giải thích cơ chế → đối chiếu output → giới hạn/kết luận
```

Ví dụ tốt:

```text
Reward gap tăng từ A lên B, nhưng chosen reward giảm C. Điều này cho thấy gap chủ yếu
đến từ rejected bị hạ nhanh hơn, phù hợp likelihood displacement. NB4 vẫn cho DPO thắng
X/8, nên behavior có cải thiện nhưng không thể kết luận chosen absolute likelihood tốt hơn.
```

## 12. Bonus +6 — NB5 merge, GGUF và deploy

### Việc làm

```bash
make deploy
```

### Kiểm tra

- Merge adapter đúng thứ tự.
- Có `gguf/*.gguf` với Q4_K_M.
- File dưới 5 GB.
- llama-cpp-python load đúng filename.
- Smoke prompt tiếng Việt sinh output mạch lạc.

### Bằng chứng

- `06-gguf-smoke.png` chứa cả dòng load filename và generated tokens.

### Điều phải hiểu

- Merge adapter khác với quantization.
- Q4_K_M đổi trade-off size/speed/quality như thế nào.
- Vì sao artifact deploy được quan trọng hơn chỉ có training checkpoint.

## 13. Bonus +8 — NB6 benchmark

### Việc làm

```bash
make bench
```

### Benchmark cần báo cáo

- IFEval.
- GSM8K.
- MMLU sampled.
- AlpacaEval-lite.

### Kiểm tra

- `data/eval/benchmark_results.json` tồn tại.
- Biểu đồ có SFT-only và SFT+DPO trên bốn benchmark.
- Mỗi bar có absolute score.
- Mỗi cặp có delta annotation.
- Reflection §7 phân tích tối thiểu 150 từ.

### Cách giải thích alignment tax

- IFEval/AlpacaEval tăng: instruction following/preference alignment có thể tốt hơn.
- GSM8K/MMLU giảm: có thể là alignment tax, formatting mismatch hoặc catastrophic forgetting; cần phân biệt.
- MMLU gần như giữ nguyên: factual knowledge được bảo toàn tương đối.
- NB4 judge và AlpacaEval-lite trái chiều: xem lại judge bias, prompt coverage và sample size.

### Bằng chứng

- `07-benchmark-comparison.png`.

## 14. Bonus +6 — β-sweep

### Trước khi chạy: viết hypothesis

Ví dụ:

- β=0.05 có thể tạo gap lớn hơn và thay đổi behavior mạnh hơn, nhưng dễ drift.
- β=0.5 có thể giữ model gần reference hơn nhưng cải thiện preference yếu.
- β=0.1 có thể là điểm cân bằng tốt nhất cho dataset/tier này.

### Việc làm

```bash
make beta-sweep
```

### Quy tắc thực nghiệm

- Cùng base/SFT adapter.
- Cùng dataset slice và seed.
- Cùng learning rate, epochs, batch và max length.
- Chỉ thay β.
- Không ghi đè baseline `adapters/dpo/`.
- So sánh cả reward gap, win-rate, output length và failure mode.

### Artifact

- `adapters/dpo-b0.05/`.
- `adapters/dpo-b0.10/`.
- `adapters/dpo-b0.50/`.
- `submission/screenshots/bonus-beta-sweep.png`.
- Bảng β trong Reflection §5.
- Phân tích ít nhất 100 từ.

### Lưu ý quan trọng

Không chọn β chỉ vì reward gap lớn nhất. β tốt nhất là β tạo behavior tốt nhất trên evaluation mà không gây length hacking, over-refusal hoặc regress benchmark quá mức.

## 15. Bonus dự phòng và nâng chất lượng

### Hugging Face Hub +5

Push **DPO adapter**, không cần push base weights:

```bash
huggingface-cli upload <user>/lab22-dpo-vn ./adapters/dpo
```

Model card phải có:

- Base model.
- SFT/preference datasets và licenses.
- Với SFT dataset đã chọn, nói rõ dataset page chưa nêu license/dataset card tại thời điểm sử dụng; không gán một license suy đoán.
- LoRA/DPO hyperparameters.
- Hardware/tier.
- Evaluation results.
- Intended use, out-of-scope use và limitations.
- Link GitHub repo.

### W&B +2

- Public run link.
- Log training loss, chosen reward, rejected reward, reward margin và learning rate.
- Tên run chứa tier, model, β và seed.

### Cross-judge +4

- Cùng outputs, chạy hai judge độc lập.
- Báo disagreement rate.
- Liệt kê prompt hai judge bất đồng.
- Có manual adjudication và giải thích rubric ưu tiên.

### GGUF release +3

- Publish ít nhất Q4_K_M và Q5_K_M.
- Ghi file size, checksum, RAM ước lượng và prompt smoke.

## 16. Research extension sau khi đã khóa điểm

Các phương pháp mới như length-normalized DPO, Robust DPO hoặc SimPO có giá trị học thuật nhưng không nằm trong rubric chính. Chỉ làm chúng trong thư mục riêng sau khi core và +20 chắc chắn.

Gợi ý ablation:

| Run | Loss | Câu hỏi |
|---|---|---|
| A | DPO sigmoid | Baseline rubric |
| B | DPO length-normalized | Có giảm length bias không? |
| C | Robust DPO | Có ổn định hơn với preference label nhiễu không? |
| D | SimPO | Reference-free objective có hiệu quả hơn không? |

Không thay baseline của lab bằng variant nghiên cứu; giữ artifacts và configs tách riêng.

## 17. Screenshot checklist an toàn

Tài liệu gọi là “minimum 6 shots” nhưng liệt kê thực tế bảy ảnh. Để tránh tranh luận, nộp đủ cả bảy:

- [ ] `01-setup-gpu.png` — GPU/CUDA/VRAM.
- [ ] `02-sft-loss.png` — SFT loss curve.
- [ ] `03-dpo-reward-curves.png` — chosen, rejected và gap.
- [ ] `04-side-by-side-table.png` — tám prompt × hai model.
- [ ] `05-judge-output.png` hoặc `05-manual-rubric.png`.
- [ ] `06-gguf-smoke.png` — NB5 bonus.
- [ ] `07-benchmark-comparison.png` — NB6 bonus.
- [ ] `bonus-beta-sweep.png` — β-sweep.

Quy tắc ảnh:

- Crop chặt và đọc được chữ.
- Có tiêu đề, legend và trục khi là biểu đồ.
- Không có API key/token.
- Không dùng ảnh toàn màn hình nếu artifact chỉ chiếm một góc.
- Reflection phải map rõ ảnh nào chứng minh criterion nào.

## 18. Artifact tree cuối cùng

```text
adapters/
├── sft-mini/
│   └── adapter_config.json
├── dpo/
│   ├── adapter_config.json
│   └── dpo_metrics.json
├── dpo-b0.05/
├── dpo-b0.10/
└── dpo-b0.50/

data/
├── pref/
│   └── train.parquet
└── eval/
    ├── prompts.json
    ├── side_by_side.jsonl
    ├── judge_results.json
    └── benchmark_results.json

gguf/
└── *Q4_K_M.gguf

notebooks/
├── 01_sft_mini.ipynb
├── 02_preference_data.ipynb
├── 03_dpo_train.ipynb
├── 04_compare_and_eval.ipynb
├── 05_merge_deploy_gguf.ipynb
└── 06_benchmark.ipynb

submission/
├── REFLECTION.md
└── screenshots/
    ├── 01-setup-gpu.png
    ├── 02-sft-loss.png
    ├── 03-dpo-reward-curves.png
    ├── 04-side-by-side-table.png
    ├── 05-judge-output.png
    ├── 06-gguf-smoke.png
    ├── 07-benchmark-comparison.png
    └── bonus-beta-sweep.png
```

Notebook `.ipynb` phải giữ output cells. Nếu dùng Jupytext, giữ cả `.py` source và `.ipynb` executed.

## 19. Trình tự chạy đề xuất

### Pass 1 — Core

```bash
make smoke
make sft
make data
make dpo
make eval
make verify
```

Sau mỗi lệnh:

1. Kiểm tra artifact.
2. Đọc output/failure mode.
3. Chụp screenshot.
4. Ghi số liệu vào Reflection ngay.
5. Chỉ chuyển bước khi checkpoint đạt.

### Pass 2 — Bonus đủ +20

```bash
make deploy
make bench
make beta-sweep
make verify
```

### Pass 3 — Submission QA

```bash
make verify
git status
git diff --check
```

Kiểm tra thủ công:

- Notebook có output.
- Reflection không còn `<...>` hoặc `_Answer here_`.
- Không có secret.
- Repo public.
- README cá nhân có link nhanh tới Reflection, screenshots, metrics, HF model nếu có.

## 20. Lịch làm hợp lý

| Giai đoạn | Thời gian dự kiến |
|---|---:|
| Đọc README/rubric + setup/smoke | 20–30 phút |
| NB1 SFT | 10–20 phút |
| NB2 data | 5–10 phút |
| NB3 DPO | 15–35 phút |
| NB4 evaluation | 15–30 phút |
| Reflection core + verify | 30–45 phút |
| NB5 deploy | 20–45 phút |
| NB6 benchmark | 30–90 phút |
| β-sweep | 45–120 phút |
| Final QA/screenshots | 20–30 phút |

Không đợi đến cuối mới viết Reflection; số liệu và interpretation dễ bị quên hoặc nhầm giữa các run.

## 21. Failure recovery

### CUDA OOM

1. Restart runtime để giải phóng fragmentation.
2. Giảm `max_length`: 512 → 384 → 256.
3. Giữ batch=1.
4. Tăng gradient accumulation: 8 → 16.
5. Nếu đang BigGPU, hạ về T4/3B.

### Reward columns không có trong logs

- Kiểm tra TRL version đúng pin của repo.
- Không nâng dependency tùy tiện giữa run.
- Kiểm tra tên columns trong `trainer.state.log_history` trước khi plot.

### Chosen reward giảm nhưng gap tăng

- Không sửa/che biểu đồ.
- Gọi đúng tên likelihood displacement.
- Đối chiếu NB4 và benchmark để xem behavior có thật sự tốt hơn không.
- Có thể dùng β-sweep để kiểm tra mức độ drift.

### Judge mặc định toàn tie

- Manual mode của notebook chỉ là placeholder.
- Phải tự điền verdict theo rubric hoặc cấu hình API judge.

### GGUF lỗi tied weights

- Làm theo hướng dẫn repo trước merge; không sửa model tùy tiện nếu chưa lưu adapter backup.

### `make verify` pass nhưng chưa chắc max điểm

`verify.py` chỉ yêu cầu tối thiểu ba screenshot và không bắt buộc NB5/NB6. Max điểm phải đối chiếu `rubric.md`, không chỉ dựa vào exit code 0.

## 22. Checklist nộp bài cuối cùng

### Core 100

- [ ] SFT adapter đúng r=16/alpha=32.
- [ ] SFT loss curve có xu hướng giảm.
- [ ] Có SFT sample generation.
- [ ] Preference parquet đúng schema.
- [ ] In ba preference examples hợp lệ.
- [ ] DPO adapter riêng biệt.
- [ ] Reward gap plot có chosen/rejected/gap.
- [ ] Reflection phân tích riêng hai reward trajectories.
- [ ] Bảng tám prompt gồm 4 helpfulness + 4 safety.
- [ ] Win/loss/tie thật, không còn manual placeholder.
- [ ] Setup/pipeline tái lập được.
- [ ] Reflection §3 và §6 đều ≥150 từ.
- [ ] `make verify` exit 0.

### Bonus +20

- [ ] NB5 GGUF + llama.cpp smoke.
- [ ] NB6 benchmark JSON + four-bar plot + Reflection §7.
- [ ] β-sweep đủ ba beta + plot + ≥100 từ phân tích.

### Submission

- [ ] Repo public.
- [ ] Executed notebooks giữ output cells.
- [ ] Đủ bảy ảnh chính và ảnh β-sweep.
- [ ] Không có API key/token.
- [ ] GitHub README link đến Reflection và artifacts quan trọng.
- [ ] URL repo được submit vào LMS trước deadline.
- [ ] Repo giữ public cho tới khi công bố điểm.

## 23. Một phút demo

Chuẩn bị lời nói ngắn theo cấu trúc:

1. **Problem:** SFT học imitation nhưng chưa trực tiếp học preference.
2. **Method:** DPO dùng chosen/rejected và reference log-ratio; β kiểm soát trade-off.
3. **Evidence:** cho xem SFT loss, ba DPO reward curves và side-by-side table.
4. **Result:** nói win/loss/tie và benchmark deltas bằng số thật.
5. **Caveat:** nêu likelihood displacement/alignment tax/length bias nếu xuất hiện.
6. **Shipping:** mở GGUF smoke hoặc HF model card.

Một bài max điểm không cần mọi metric đều tăng. Bài mạnh là bài có pipeline đúng, bằng chứng đầy đủ, giải thích trung thực và biết phân biệt “training signal đẹp” với “behavior thực sự tốt hơn”.

