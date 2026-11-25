# Paper 1 (`ucai-verify`): การทวนสอบลูกผสมเชิงกำหนดก่อนเข้าโมเดลเพื่อป้องกันอ้างอิงผีและการสลับตัวเลขข้ามเอกสารในงานเขียนวิทยาศาสตร์โดย AI

> **ชื่อเรื่องภาษาอังกฤษ:** *`ucai-verify`: Deterministic-First Hybrid Verification Against Ghost Citations and Cross-Artifact Numeric Misattribution in AI-Generated Scientific Writing*
> **ผู้วิจัย:** นพดล ชัยวิสูตร (Noppadon Chaiwisut) — นักวิจัยอิสระ ประเทศไทย (`Independent Researcher, Thailand`)
> **ORCID:** `0009-0005-2881-8458` | **อีเมล:** `noppadon.chaiwisut@gmail.com`
> **รุ่น / วันที่:** Version 0.2.0 (Tier-2 Empirical Release) | พฤศจิกายน 2026
> **หมายเหตุการทำซ้ำ:** โค้ด (`MIT`) & `UCAI-CitationAudit-Bench-v1` (`CC BY 4.0`); Colab: `12T9Zl8y2XJqF1gnODZ37yJPy_8xHQ_ME`; Zenodo DOI: *Pending v0.2.0 Release*

---

## บทคัดย่อ (Abstract)

เอเจนต์ช่วยเขียนงานวิจัยอัตโนมัติและโมเดลภาษาขนาดใหญ่ (LLM) มักสร้างต้นฉบับที่อ่านลื่นไหลแต่แฝงความคลาดเคลื่อนร้ายแรง 2 ประเภท คือ **อ้างอิงผีที่ไม่มีอยู่จริง (Non-existent ghost citations)** และ **การสลับตัวเลขข้ามเอกสาร (Cross-artifact numeric misattribution)** ซึ่งนำตัวเลขจริงจากเปเปอร์หนึ่งไปแปะให้เปเปอร์จริงอีกฉบับอย่างแนบเนียน วิธีป้องกันที่นิยมเชื่อกันคือการใช้ LLM ตรวจกันเอง (*Self-Critique* หรือ *LLM-as-a-Judge*)

เมื่อประเมิน 5 รูปแบบการทวนสอบบน **`UCAI-CitationAudit-Bench-v1`** (`N = 120` ข้อสมดุล 5 คลาส อ้างอิงคลังเปิด `168` PDFs) เราพบกับดักสำคัญคือ **Pure-LLM False-Positive Trap**: หากไม่มีการยึดโยงกับฐานข้อมูลบรรณานุกรมสดและข้อความเต็ม, *Pure Self-Critique* และ *Pure LLM-as-a-Judge* จะปรักปรำเปเปอร์จริงปี 2024–2026 ว่าแต่งปลอมถึง **83.3% (`20/24`)** และ **91.7% (`22/24`)** ตามลำดับ

เพื่อแก้ปัญหานี้ เราเสนอ **`ucai-verify` (v0.2.0)** ระบบทวนสอบลูกผสม 2 ขั้นตอนแบบเชิงกำหนดก่อนเข้าโมเดล ภายใต้การกำกับดูแล **`L2 HITL`**:
- **Stage 1 (`$0` Token):** ตรวจสอบตัวตนบรรณานุกรมข้าม 3 ฐานข้อมูลสด (`arXiv` แบบรู้เวอร์ชัน v1/v2, `Crossref`, `OpenAlex`) ร่วมกับการจับคู่ตัวเลขข้ามไฟล์ PDF รายหน้าด้วย `PyMuPDF` ทำ **Precision `100.0%`**, **FPR `0.0%` (`0.15 s/ข้อ`)** และกรองข้อสอบออกได้ **`60.0%` (`72/120`)** โดยไม่ต้องเรียกโมเดลประสาทเทียม
- **Stage 2:** ประเมินทิศทางความหมาย (Polarity) เฉพาะ **`48` ข้อ** ที่รอดจาก Stage 1 เทียบกับท่อนหลักฐานจริงจาก PDF:
  - **โหมดออฟไลน์ (`DeBERTa-v3-base`):** ได้ **`F1 = 0.9388`** (`95% CI [0.9022, 0.9694]`, Recall `95.83%`, ค่า API `$0`)
  - **โหมดคลาวด์จำกัดขอบเขต (`Hybrid-Scoped-Gemini`):** ได้ **`F1 = 0.9796`** (`95% CI [0.9592, 0.9951]`), **Recall `100.0%` (`96/96`)**, ลดการเรียก LLM API **`60.0%`**, และชนะ *Pure LLM-as-a-Judge* **`21 ต่อ 0`** (`p = 1.0 × 10⁻⁶`)

---

## 1. บทนำ (Introduction)

แม้เครื่องมือวิจัยอัตโนมัติจะเร่งการร่างบทความได้มาก แต่สถิติปี 2025 พบอ้างอิงผีอย่างน้อย **`146,932` รายการ** (หลุดตีพิมพ์ **`85.3%`**) และงานประชุม ML ชั้นนำพบอ้างอิงหลอน **`18.7%–26.2%`** ขณะที่ความลื่นไหลของภาษายิ่งทำให้มนุษย์จับข้ออ้างไร้หลักฐานได้ยากขึ้น

ระหว่างการสังเคราะห์โครงการของเราเอง เราพบกรณีจริงของ *การสลับตัวเลขข้ามเอกสาร*: ร่าง LLM อ้างถึง *ScienceAgentBench* (`arXiv:2410.05080`) ซึ่งเป็นเปเปอร์จริงใน ICLR 2025 แต่กลับใส่คะแนนความถูกต้อง **`92.2%`** ซึ่งแท้จริงเป็นตัวเลขของ *AutoSDT* (`arXiv:2506.08140`, หน้า 2) ทั้งที่ค่าจริงของ *ScienceAgentBench* คือ `84.0%` execution rate และ `22.9%` success rate

**คุณูปการหลัก 4 ประการของ Paper 1:**
1. **สถาปัตยกรรม Deterministic-First Hybrid (`ucai-verify v0.2.0`)** รองรับทั้งโหมดออฟไลน์ (`DeBERTa-v3-base`) และโหมดคลาวด์จำกัดขอบเขต
2. **ชุดทดสอบมาตรฐาน `UCAI-CitationAudit-Bench-v1`** (`120` ข้อ สมดุล 5 คลาส จาก 12 เปเปอร์ปี 2024–2026 ในคลัง 168 PDFs)
3. **การพิสูจน์เชิงประจักษ์ถึง Pure-LLM False-Positive Trap** (`83.3%–91.7%` FPR)
4. **ความเหนือกว่าทางสถิติอย่างมีนัยสำคัญ** (`F1 = 0.9388` ออฟไลน์ / `0.9796` Scoped Gemini, `p = 1.0 × 10⁻⁶`)

---

## 2. ระเบียบวิธี (`ucai-verify` v0.2.0)

- **Stage 1A (Version-Aware Citation Cascade):** ตรวจ `arXiv Atom API` (รองรับการเช็ก `v1` เมื่อผู้เขียนเปลี่ยนชื่อเรื่องใน `v2`) และ `Crossref REST API` หากไม่มี ID จะค้นด้วยชื่อเรื่องผ่าน Crossref/OpenAlex ด้วยสูตร `Sim(Ta, Tb) = max(SeqRatio, Jaccard)` ร่วมกับเงื่อนไขนามสกุลผู้แต่ง (`≥ 0.50`) และปี (`±1`) ออกสถานะ `VERIFIED_EXACT`, `METADATA_DRIFT`, `IDENTIFIER_HIJACKED` (`Sim < 0.45`), หรือ `SUSPECTED_FAKE`
- **Stage 1B (Page-Indexed Cross-PDF Numeric & Lexical Audit):** สกัดตัวเลข (ยกเว้นปีและ ID) ปรับรูปมาตรฐาน 3 แบบ: ตัวคั่นหลักพัน (`146,932 ↔ 146932`), การตัดคำขึ้นบรรทัดใหม่, และร้อยละ/ทศนิยม (`87.4% ↔ 0.874`) หากไม่พบเลขในเอกสารที่อ้าง จะสแกนอีก `167` ไฟล์ที่เหลือเพื่อจับ `CROSS_ARTIFACT_MISATTRIBUTED` พร้อมระบุเลขหน้าต้นทางจริง `(Dm', page k)`
- **Stage 2 (Grounded Polarity Verification):** เฉพาะข้อที่ผ่าน Stage 1 เท่านั้นจึงจะเข้า Stage 2 เทียบข้อความอ้างอิงกับท่อนหลักฐานจาก PDF (`≤ 900` ตัวอักษร) เลือกได้ 2 โหมดคือ **Mode A (`Hybrid-Local-NLI` — `$0` ออฟไลน์)** และ **Mode B (`Hybrid-Scoped-Gemini`)** ภายใต้มาตรฐาน NISO JATS/CREC และ `L2 HITL`

---

## 3. การตั้งค่าการทดลอง (Experimental Setup)

- **ชุดทดสอบ `UCAI-CitationAudit-Bench-v1` (`N = 120`, `CC BY 4.0`):** แบ่งเป็น 5 คลาส คลาสละ `24` ข้อ:
  1. `VALID` (`n=24`) — ข้อมูลจริง ตัวเลขจริง ทิศทางตรง
  2. `FAKE_AUTHOR_TITLE` (`n=24`) — ชื่อเรื่อง/ผู้แต่ง/DOI ปลอม
  3. `BAD_DOI_ARXIV_ID` (`n=24`) — เปเปอร์จริงแต่ใส่ DOI/arXiv ID ผิดหรือถูกสวมสิทธิ์
  4. `NUMERIC_MISATTRIBUTION` (`n=24`) — เปเปอร์จริงแต่สลับตัวเลขจากเปเปอร์อื่นหรือเพี้ยนระดับสิบเท่า
  5. `POLARITY_REVERSAL` (`n=24`) — เปเปอร์จริง เลขจริง แต่สรุปผลกลับทิศทาง
- **คู่เทียบ 5 ระบบบน Google Colab (`120/120` สำเร็จ 0 API errors):** ใช้ Bootstrap `1,000` รอบ (seed 42) และ Exact McNemar test

---

## 4. ผลการทดลองและวิเคราะห์ความผิดพลาด (Results and Analysis)

### ตารางที่ 1: ผลลัพธ์รวมบน `UCAI-CitationAudit-Bench-v1` (`N = 120`)

| รูปแบบการทวนสอบ | Acc. `↑` | Prec. `↑` | Recall `↑` | F1 [95% Bootstrap CI] `↑` | FPR `↓` | LLM Calls | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ucai-verify` Stage 1 (`$0`)** | `0.8000` | **`1.0000`** | `0.7500` | `0.8571 [0.8000, 0.9111]` | **`0.0000` (`0/24`)** | **`0`** | **`0.15 s`** |
| **`Hybrid — Local NLI` (`$0`)** | `0.9000` | `0.9200` | `0.9583` | `0.9388 [0.9022, 0.9694]` | `0.3333` (`8/24`) | **`0`** | `2.98 s` |
| **`Hybrid — Scoped Gemini`** | **`0.9667`** | `0.9600` | **`1.0000`** | **`0.9796 [0.9592, 0.9951]`\*\*\*** | `0.1667` (`4/24`) | `48` (`-60%`) | `3.03 s` |
| `Pure Self-Critique` (`120`) | `0.8083` | `0.8230` | `0.9688` | `0.8900 [0.8431, 0.9309]` | `0.8333` (`20/24`) | `120` | `9.07 s` |
| `Pure LLM-as-a-Judge` (`120`) | `0.7917` | `0.8087` | `0.9688` | `0.8815 [0.8276, 0.9266]` | `0.9167` (`22/24`) | `120` | `7.07 s` |

- **ผลรายคลาสและวิเคราะห์จุดพลาด:** Stage 1 จับ `FAKE_AUTHOR_TITLE` และ `BAD_DOI_ARXIV_ID` ได้ **`100.0%`**, จับ `NUMERIC_MISATTRIBUTION` ได้ `83.3%` (`20/24`) ส่วน 4 ข้อที่หลุดเป็นเลขที่บังเอิญไปตรงกับเลขอื่น (เช่น เลขหน้าบรรณานุกรม) ซึ่ง Stage 2 (`Scoped Gemini`) เก็บคืนครบ `100.0%` และใน `POLARITY_REVERSAL` Stage 2 ยกความแม่นยำจาก `16.7%` ขึ้นเป็น `91.7%` (Local NLI) และ **`100.0%`** (Scoped Gemini)
- **วิเคราะห์ False Positive ที่เหลือ (`4/24` ข้อ `VALID`):** เกิดจาก (1) ข้อความที่มีตัวเลขหลายค่าข้ามหน้าเกินหน้าต่าง `900` ตัวอักษร (2 ข้อ) และ (2) ความต่างการย่อชื่อรุ่นโมเดล เช่น `GPT-5` vs `GPT-5.4` (2 ข้อ) ซึ่งระบบ `L2 HITL` ส่งพิกัดหน้า PDF ให้ผู้เขียนตรวจสอบในไม่กี่วินาที

---

## 5–6. ข้อจำกัด จริยธรรม และบทสรุป

Stage 1B พึ่งพาข้อความดิจิทัลจาก `PyMuPDF` (เอกสารสแกนที่ไม่มี OCR ต้องใช้ตัวอ่านภาพอย่าง Docling) และในด้านจริยธรรม ผลแจ้งเตือนทั้งหมดเป็นหลักฐานช่วยตรวจภายใต้ **`L2 HITL`** ห้ามใช้ลงโทษทางวิชาการอัตโนมัติ โดยสรุป `ucai-verify v0.2.0` พิสูจน์ว่าการตรวจโดย LLM ล้วนแก้ปัญหาอ้างอิงหลอนไม่ได้ (`FPR` สูงถึง `91.7%`) แต่การยึดโยงฐานข้อมูลสดและตัวเลขใน PDF ก่อนเข้าโมเดลทำ **`F1 = 0.9388` (`$0` ออฟไลน์)** และ **`F1 = 0.9796` (ลด API `60%`)**

---

## แหล่งอ้างอิง (References)

1. Lu, Chris and Lu, Cong and Lange, Robert Tjarko and Foerster, Jakob and Clune, Jeff and Ha, David (2024). *The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery*. https://arxiv.org/abs/2408.06292
2. Gottweis, Juraj and Weng, Wei-Hung and Daryin, Alexander and Tu, Tao and Palepu, Anil and others (2025). *Towards an AI Co-Scientist*. https://arxiv.org/abs/2502.18864
3. Mitchener, Ludovico and Yiu, Angela and Chang, Benjamin and White, Andrew D. (2025). *Kosmos: An AI Scientist for Autonomous Discovery*. https://arxiv.org/abs/2511.02824
4. Skarlinski, Michael D. and Cox, Sam and Laurent, Jon M. and Braza, James D. and White, Andrew D. (2024). *Language Agents Achieve Superhuman Synthesis of Scientific Knowledge*. https://arxiv.org/abs/2409.13740
5. Fischer, Markus and Meiler, Jens and Elgeti, Matthias (2026). *MuninAI: A Self-Hosted Agentic Framework for Independent Research Groups*. https://arxiv.org/abs/2610.23480
6. Zhao, Zhenyue and Wang, Yihe and Stuart, Toby and De Vaan, Mathijs and Ginsparg, Paul and Yin, Yian (2026). *LLM Hallucinations in the Wild: Large-Scale Evidence from Non-Existent Citations*. https://arxiv.org/abs/2605.07723
7. Russinovich, Mark and Siva Kumar, Ram Shankar and Salem, Ahmed (2026). *Phantom References: Hallucinated Citations That Survive Peer Review at Top-Tier Conferences*. https://arxiv.org/abs/2607.00738
8. Liu, Nelson F. and Zhang, Tianyi and Liang, Percy (2023). *Evaluating Verifiability in Generative Search Engines*. https://aclanthology.org/2023.findings-emnlp.467.pdf
9. Li, Yifei and Yue, Xiang and Liao, Zeyi and Sun, Huan (2024). *AttributionBench: How Hard is Automatic Attribution Evaluation?*. https://aclanthology.org/2024.findings-acl.886.pdf
10. Chen, Ziru and Chen, Shijie and Ning, Yuting and Zhang, Qianheng and Wang, Boshi and Sun, Huan (2025). *ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery*. https://arxiv.org/abs/2410.05080
11. Li, Yizhe and Moussa, Hanyuan Ning and Chen, Ziru and Sun, Huan (2025). *AutoSDT: Scaling Data-Driven Discovery Toward Open Co-Scientists*. https://arxiv.org/abs/2506.08140
12. Wadden, David and Lin, Shanchuan and Lo, Kyle and Wang, Lucy Lu and van Zuylen, Madeleine and Cohan, Arman and Hajishirzi, Hannaneh (2020). *Fact or Fiction: Verifying Scientific Claims*. https://aclanthology.org/2020.emnlp-main.609.pdf
13. Wadden, David and Lo, Kyle and Wang, Lucy Lu and Cohan, Arman and Beltagy, Iz and Hajishirzi, Hannaneh (2022). *MultiVerS: Improving Scientific Claim Verification with Weak Supervision and Full-Document Context*. https://aclanthology.org/2022.findings-naacl.6.pdf
14. Min, Sewon and Krishna, Kalpesh and Lyu, Xinxi and Lewis, Mike and Yih, Wen-tau and Koh, Pang Wei and Iyyer, Mohit and Zettlemoyer, Luke and Hajishirzi, Hannaneh (2023). *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*. https://aclanthology.org/2023.emnlp-main.741.pdf
15. Gao, Tianyu and Yen, Howard and Yu, Jiatong and Chen, Danqi (2023). *Enabling Large Language Models to Generate Text with Citations*. https://aclanthology.org/2023.emnlp-main.398.pdf
16. Sadat, Mobashir and Caragea, Cornelia (2024). *MSciNLI: A Diverse Benchmark for Scientific Natural Language Inference*. https://aclanthology.org/2024.naacl-long.90.pdf
17. Tang, Liyan and Laban, Philippe and Durrett, Greg (2024). *MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents*. https://aclanthology.org/2024.emnlp-main.499.pdf
18. Khajavi, Khashayar and Sadeghi, Shaghayegh and Adhikari, Rise and Tessier, Alexander (2026). *CITECHECK: Retrieval-Grounded Detection of LLM Citation Hallucinations in Scientific Text*. https://arxiv.org/abs/2605.27700
19. Aly, Rami and Guo, Zhijiang and Schlichtkrull, Michael and Thorne, James and Vlachos, Andreas and Christodoulopoulos, Christos and Cocarascu, Oana and Mittal, Arpit (2021). *FEVEROUS: Fact Extraction and VERification Over Unstructured and Structured Information*. https://arxiv.org/abs/2106.05707
20. Lu, Xinyuan and Pan, Liangming and Liu, Qian and Nakov, Preslav and Kan, Min-Yen (2023). *SCITAB: A Challenging Benchmark for Compositional Reasoning and Claim Verification on Scientific Tables*. https://aclanthology.org/2023.emnlp-main.483.pdf
21. Venktesh, V. and Anand, Abhijit and Anand, Avishek and Setty, Vinay (2024). *QuanTemp: A Real-World Open-Domain Benchmark for Fact-Checking Numerical Claims*. https://arxiv.org/abs/2403.17169
22. Wang, H. and Ren, Y. and Bommireddypalli, S. and Chen, S. and Prabhudesai, A. and Zhou, R. and Baral, E. and Koehn, P. (2026). *SciTaRC: A Plan-Annotated Scientific Tabular QA Benchmark for Language Reasoning and Complex Computation*. https://arxiv.org/abs/2603.08910
23. B\"oschen, Ingmar (2026). *Extraction of Tabulated Statistical Results with tableParser*. https://arxiv.org/abs/2603.19756
24. Hendricks, Ginny and Tkaczyk, Dominika and Lin, Jennifer and Feeney, Patricia (2020). *Crossref: The Sustainable Source of Community-Owned Scholarly Metadata*. https://doi.org/10.1162/qss_a_00022
25. Priem, Jason and Piwowar, Heather and Orr, Richard (2022). *OpenAlex: A Fully-Open Index of Scholarly Works, Authors, Venues, Institutions, and Concepts*. https://arxiv.org/abs/2205.01833
26. Kinney, Rodney and Anastasiades, Chloe and Beltagy, Iz and Cohan, Arman and Downey, Doug and Etzioni, Oren and others (2025). *The Semantic Scholar Open Data Platform*. https://arxiv.org/abs/2301.10140
27. Chen, Hui and Xiong, Miao and Lu, Yujie and Han, Wei and Hooi, Bryan (2025). *MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research*. https://arxiv.org/abs/2505.19955
28. National Information Standards Organization (2024). *ANSI/NISO Z39.96-2024, JATS: Journal Article Tag Suite, Version 1.4*. https://doi.org/10.3789/ansi.niso.z39.96-2024
29. National Information Standards Organization (2024). *NISO RP-45-2024: Communication of Retractions, Removals, and Expressions of Concern (CREC)*. https://doi.org/10.3789/niso-rp-45-2024
30. Li, L. and Xiong, J. and Zhu, C. and Yu, R. and Lu, Z. (2026). *LLM-as-a-Reviewer: Benchmarking Their Ability, Divergence, and Prompt Injection Resistance as Paper Reviewers*. https://arxiv.org/abs/2605.25415
31. Loc, N. P. P. and Viet, T. H. L. and Nguyen, T. and Nguyen, B. T. (2026). *PRISM: A Multi-Dimensional Benchmark for Evaluating LLM Peer Reviewers*. https://arxiv.org/abs/2605.26730
32. Livathinos, Nikolaos and Auer, Christoph and Lysak, Maksym and Nassar, Ahmed and Dolfi, Michele and Vagenas, Panos and others (2024). *Docling Technical Report*. https://arxiv.org/abs/2408.09869
