### 深海牧场边界：基于冷泉喷口物质扩散的生态临界探索  
**进展报告**  

**报告日期：** 2024年12月29日  
**负责人：** 周陈序  
**研究团队：** 蔡睿，龚厚霖，余逸康，曾嘉  
**指导老师：** 肖湘，高晓沨，赵维殳  

---

### **一、项目背景与目标**

**背景：**  
冷泉是深海生态系统的重要组成部分，其喷口物质扩散形成的特殊环境对海底生物群落的演化和分布有着重要影响。研究冷泉喷口物质的扩散与生态边界的关系，不仅可以帮助理解深海牧场的形成机制，还可为海底资源保护提供理论支持。

**项目目标：**  
1. **海底地形建模：**  
   - 利用深海探测视频和图像，完成冷泉区域海底生境的地形图拼接和补全。  
   - 结合算法与人工标注方法，完成冷泉周边区域的地形建模和补全工作。  
2. **生态研究：**  
   - 研究冷泉环境对海底生物的影响，尤其是生物突变和群落分布的特点。  
   - 重点探索冷泉生态边界的识别方法及其发育程度的判断。  
   - 理想目标为通过上层海水理化性质推断冷泉生态状态，并建立区域模型。

---

### **二、研究进展**

#### 1. 海底生境地图重建  
- **工作内容：**  
  - 使用709航次2-27获取的深海探测数据，结合高清摄像头影像，进行冷泉区域海底地图的拼接和建模。  
  - 当前完成至第17站点数据处理，完成约40%的地图拼接工作。  
- **关键进展：**  
  - 临近冷泉喷口区域，发现贝类突变现象，表明喷口附近的特殊环境可能对生物群落产生显著影响。  
  - 拼接算法已基本稳定，但因摄像头参数（如焦距）缺失，导致图像视觉畸变问题未完全解决。
  - 航迹未覆盖完整冷泉区，存在空洞，可通过其它航次补充或采用算法补全。

#### 2. 冷泉生物识别  
- **工作内容：**  
  - 采用YOLO模型对拼接地图中的生物群落进行自动识别，探索生物分布与冷泉环境边界的关系。  
- **当前问题：**  
  - 现有识别模型的准确率较低，部分海底生物（如贝类）因图像质量问题无法准确分类。  
  - 数据集不足，模型训练不充分。  
- **后续计划：**  
  - 寒假期间通过人工标注方法制作高清图片训练集，优化模型，提升识别准确率。  

#### 3. 技术与流程优化  
- 摄像头畸变矫正：计划使用坐标纸校准拍摄设备的畸变参数，并可能在后续工作中应用矫正算法。  
- 工作流程整理：编制标准化操作流程文档，包括数据获取、图像处理、模型训练和结果分析的各环节规范。  

---

### **三、后续工作计划**

1. **近期任务：**  
   - 完成海底地图剩余拼接任务，完成超过40%以外的未完成部分。  
   - 整理冷泉区域的贝类突变现象数据，初步建立生物特征数据集。  
   - 对颜色差异较大的区域开展独立标注工作，并进行初步分析。  

2. **寒假安排：**  
   - 举办标注方法教学小组会议，讲解深海生物标注与冷泉区域案例研究的方法。  
   - 在会议前完成典型样例的筛选和准备工作，并邀请指导老师（赵维殳）协助完成识别任务。  
   - 人工标注完成后，优化YOLO模型的训练，验证模型性能提升效果。

3， **海水理化性质分析**
    - 数据处理中
    - 数据处理完成后可据此建立理化性质模型，协同生物分布分析冷泉边界与生态影响

3. **技术文档编写：**  
   - 整理本项目的技术流程和规范，包括数据种类、格式需求以及处理标准。  
   - 编写详细的图像拼接和标注流程文档，确保团队成员协同工作。  

---

### **四、项目最终目标**

1. **完成冷泉区域地形建模：**  
   - 通过拍摄视频拼接和算法补全相结合，构建冷泉区域完整的海底生境地形图。  
2. **冷泉生态边界探索：**  
   - 基于模型分析冷泉环境对海底生物的影响，研究冷泉区域生物的突变现象及群落分布特征，建立冷泉生态边界的识别和评价模型。  
   - 探索利用海水理化性质反推冷泉生态状态的理论框架。  

---

**报告撰写人：** 周陈序  
**报告日期：** 2024年12月29日  
