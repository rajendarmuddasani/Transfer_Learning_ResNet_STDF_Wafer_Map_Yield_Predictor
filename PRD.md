# Product Requirements Document (PRD)
# P02: Transfer Learning Test Yield Predictor

**Project ID**: P02_Transfer_Learning_Yield_Predictor  
**Category**: Semiconductor Post-Silicon Validation / Deep Learning / Transfer Learning  
**Status**: Draft for Review  
**Version**: v1.0  
**Last Updated**: 2025-12-04  
**Product Family**: Automotive MCU (TC3x, TC4x families)  
**Test Platform**: Advantest V93000 SMT8, Teradyne testers  

---

## 1. Overview

### 1.1 Executive Summary

The Transfer Learning Test Yield Predictor is a deep learning-powered computer vision platform that leverages ResNet-based transfer learning to predict final test yield from wafer map images generated from STDF test data. By applying pre-trained convolutional neural networks (ResNet-18/ResNet-50) originally trained on ImageNet to semiconductor wafer map spatial patterns, the system transforms raw die-level test results (lot, wafer, x, y, bin) into 300x300 RGB wafer map images, then classifies spatial defect patterns (edge effects, center clusters, ring patterns, quadrant failures) to predict final yield with >92% accuracy using only 5-10% of completed tests.

The platform employs sophisticated transfer learning strategies: (1) **Progressive Fine-tuning** - freeze ResNet backbone → unfreeze last block → full network adaptation, (2) **Domain Adaptation** - bridge ImageNet natural images to semiconductor wafer maps via batch normalization adaptation and feature alignment, (3) **Discriminative Learning Rates** - higher learning rates for classifier head (1e-3), lower for pre-trained layers (1e-5), and (4) **Data Augmentation** - rotations, flips, color jitter to maximize limited semiconductor training data. This enables rapid model adaptation for new products with <500 samples (vs. 10,000+ traditional training), reducing NPI cycle time by 6 months.

**Key Value Proposition**: Reduce test costs by $5M+ annually through 30-40% test time reduction via adaptive termination, enable spatial root cause analysis invisible to parametric-only methods, and accelerate new product introduction with transfer learning achieving >85% accuracy using 500 samples vs. 10,000+ for from-scratch CNN training.

### 1.2 Document Purpose

This PRD defines comprehensive requirements for designing, developing, testing, and deploying the Transfer Learning Test Yield Predictor system. It covers:
- Functional and non-functional requirements for CNN model training, transfer learning, and inference
- Data ingestion pipelines for STDF parsing, wafer map generation, and image preprocessing
- System architecture with API specifications, model serving infrastructure, and database schemas
- UI/UX requirements for wafer map visualization, model monitoring dashboards, and prediction explainability
- Security, performance, scalability, and testing strategies for production deployment
- Deployment phases (offline validation → shadow mode → assisted prediction → automated termination) with rollback procedures
- Success metrics, KPIs, and business impact validation across multiple product lines

The document serves as the single source of truth for cross-functional teams (ML Engineering, Computer Vision, Backend, Frontend, DevOps, Test Engineering, Yield Engineering, Product Engineering) throughout the development lifecycle.

### 1.3 Product Vision

**Vision Statement**: Establish the industry-leading computer vision platform for semiconductor yield prediction, combining transfer learning from massive natural image datasets with domain-specific adaptation for wafer defect pattern recognition, enabling zero-touch yield forecasting and autonomous test optimization across all automotive MCU families.

**Long-term Goals** (18-24 months):
- Deploy across 15+ product/package combinations (TC3x, TC4x families, all package variants)
- Achieve >92% yield prediction accuracy with only 5-10% test completion
- Enable 30-40% average test time reduction through confidence-based adaptive termination
- Support rapid new product onboarding with transfer learning from mature products (<500 samples vs. 10,000+ traditionally)
- Integrate multi-modal inputs (wafer maps + parametric trends + shmoo plots + spatial statistics)
- Establish as reference architecture for applying computer vision to semiconductor manufacturing

**Differentiation**:
- Transfer learning from ImageNet → semiconductor domain (vs. training from scratch)
- Progressive fine-tuning strategy: freeze backbone → unfreeze last block → full network (vs. single-phase training)
- Multi-resolution wafer map analysis: device-level (300x300) + die-level (32x32 patches) + full-wafer (1024x1024)
- Spatial pattern classification: 8+ defect types (edge, center, ring, quadrant, scratch, cluster, random, mixed)
- Domain adaptation techniques: feature alignment, batch normalization adaptation, adversarial training
- Explainability via Grad-CAM: highlight wafer regions driving yield predictions
- Real-time inference (<200ms per wafer) with GPU acceleration and ONNX optimization

---

## 2. Problem Statement

### 2.1 Current Challenges

**Challenge 1: Spatial Information Loss in Traditional ML**
- Existing bin predictors (XGBoost, RandomForest) treat test results as flat feature vectors, ignoring spatial relationships
- Die position (x, y) encoded as numeric features loses geometric structure and proximity information
- Systematic spatial failures (edge effects, center hot spots) difficult to detect without spatial modeling
- Wafer-level patterns (ring defects, quadrant variations) require manual visual inspection by yield engineers
- No automated correlation between parametric test failures and physical wafer locations

**Challenge 2: Long Training Times and Data Hunger**
- Training CNN from scratch on semiconductor data requires 50,000+ labeled wafer maps
- New product introduction (NPI) delays: must collect 6-12 months of production data before accurate models
- Product-specific models don't transfer knowledge: TC41x model useless for TC42x despite similar failure modes
- Package variant changes (BGA436 → BGA292) require complete model retraining
- Cold-start problem: no predictions available during early production ramp (PR7-PR8 milestones)

**Challenge 3: Inadequate Yield Prediction Timing**
- Current yield forecasting requires 80-100% test completion (8-12 minutes per device)
- By the time low yield detected, entire wafer/lot already processed (2-3 day latency)
- Cannot abort testing early for high-confidence FAIL predictions (wasted tester capacity)
- No mechanism to prioritize suspect devices for extended characterization
- Reactive rather than predictive: yield learning happens after-the-fact

**Challenge 4: Limited Defect Pattern Recognition**
- Manual wafer map review by yield engineers (40+ hours/month per engineer)
- Inconsistent pattern labeling: same defect called "edge effect" vs. "peripheral failure" by different engineers
- Rare defect patterns (ring, scratch, mixed-mode) require expert knowledge to identify
- No quantitative similarity scoring: "Is this wafer map similar to historical lot X123?"
- Tribal knowledge not captured in reproducible models

**Challenge 5: Multi-Modal Data Integration Gaps**
- Wafer maps analyzed separately from parametric test trends (IDDQ, Vth, Fmax)
- Shmoo plot patterns (voltage-frequency sweeps) not correlated with spatial failures
- Environmental data (temperature, humidity during test) not integrated into yield models
- No unified view combining: spatial (wafer map) + temporal (test sequence) + parametric (measurements)

### 2.2 Impact Analysis

**Business Impact**:
- **Test Cost**: $8-12M annual test spend across TC4x family products
- **Capacity Constraints**: 24 test sites × 90% utilization = continuous bottleneck, 2-week backlog during peak
- **Yield Loss**: 3-8% yield sacrificed from over-conservative test limits (spatial patterns not understood)
- **NPI Delays**: 6-month test/yield optimization per new product = missed automotive customer windows
- **Scrap Cost**: Late excursion detection = $150K-$300K scrap per event (2-3 lots affected)

**Engineering Impact**:
- **Labor**: 120+ hours/month manual wafer map review across 3 yield engineers
- **Learning Cycles**: 8-12 weeks to identify spatial root causes (edge effect, center cluster, etc.)
- **Model Training**: 4-6 weeks to retrain models for each product/package variant
- **Knowledge Loss**: Spatial pattern expertise leaves with retiring engineers (tribal knowledge)
- **Tool Fragmentation**: 5+ separate tools for wafer map viewing, STDF parsing, yield analysis, pattern recognition

**Quality Impact**:
- **DPPM**: Spatial defects escaping to customers (marginal edge dies passing test but failing in field)
- **Field Returns**: 30-50 PPM returns with spatial failure signatures not detected during manufacturing
- **Reliability**: Latent defects in marginal spatial zones (near edge, high-power regions) causing early-life failures
- **Customer Satisfaction**: Automotive OEM complaints about yield excursions and quality escapes

**Strategic Impact**:
- **Competitiveness**: Foundry competitors using AI/ML for 40% faster yield ramp
- **Scalability**: Manual processes don't scale to 100+ product variants planned (2026-2028)
- **Innovation Velocity**: Engineering bandwidth consumed by reactive firefighting vs. next-gen product development
- **AI/ML Capability**: Organization lacks computer vision expertise in semiconductor manufacturing context

### 2.3 Opportunity

**Immediate Opportunities** (0-6 months):
- **Test Time Reduction**: Predict yield after 5-10% test completion → adaptive termination → 30-40% time savings → $3-5M annual cost reduction
- **Spatial Pattern Detection**: Automated wafer map classification (8 defect types) → eliminate 100+ manual review hours/month → $180K annual labor savings
- **Early Excursion Alerts**: Real-time yield prediction per wafer (vs. per lot) → detect issues within 4 hours vs. 48 hours → 90% scrap reduction
- **Transfer Learning Acceleration**: New product models with <500 samples (vs. 10,000+) → 6-month NPI time reduction → $2M opportunity cost savings

**Medium-term Opportunities** (6-12 months):
- **Multi-Modal Fusion**: Combine wafer maps + parametric trends + shmoo patterns → >95% prediction accuracy (vs. 92% single-modal)
- **Defect Localization**: Grad-CAM visualization highlights problematic wafer regions → direct yield engineers to spatial root causes
- **Cross-Product Learning**: TC41x knowledge transfers to TC42x, TC43x → continuous improvement across portfolio
- **Adaptive Test Sequencing**: Predict spatial failure types early → route to targeted characterization → 50% debug efficiency gain

**Long-term Opportunities** (12-24 months):
- **Closed-Loop Optimization**: Wafer map predictions → SPC alerts → process adjustments → predictive yield management
- **Design-for-Manufacturability**: Feedback spatial failure patterns to design teams → improve next-gen products
- **Foundry Collaboration**: Share anonymized spatial signatures with fab → joint process optimization
- **Multi-Site Deployment**: Expand beyond TC families to other product lines (Power Management ICs, Sensors, RF)

**Strategic Opportunities**:
- **IP Creation**: Patent transfer learning techniques for semiconductor yield (10+ invention disclosures)
- **Talent Development**: Build computer vision center-of-excellence within post-silicon organization
- **Customer Differentiation**: Demonstrate AI/ML leadership to automotive OEMs (Tier-1 supplier advantage)
- **Publication**: 3-5 papers at IEDM, ISSCC, CVPR demonstrating semiconductor + computer vision innovation

---

## 3. Goals and Objectives

### 3.1 Primary Goals

**Goal 1: Build Production-Grade Transfer Learning Models**
- Train ResNet-18/ResNet-50 models with ImageNet pre-trained weights achieving >92% yield prediction accuracy
- Support 8+ spatial defect classifications: Edge Effect, Center Cluster, Ring Pattern, Quadrant Failure, Scratch/Line, Random Failure, Mixed-Mode, Normal
- Provide calibrated confidence scores (0-100%) per prediction with uncertainty quantification via Monte Carlo Dropout
- Enable early prediction with only 5-10% test completion (first 500 of 10,000 tests) maintaining >88% accuracy

**Goal 2: Implement Effective Transfer Learning Pipeline**
- Progressive fine-tuning strategy: Phase 1 (freeze backbone, train classifier) → Phase 2 (unfreeze last block) → Phase 3 (full network fine-tuning)
- Discriminative learning rates: higher LR for classifier head (1e-3), lower LR for backbone (1e-5)
- Domain adaptation via batch normalization statistics adaptation, optional adversarial domain alignment
- Data augmentation pipeline: rotations (0°, 90°, 180°, 270°), horizontal/vertical flips, color jitter, random crops
- Support rapid adaptation for new products: achieve >85% accuracy with 500 samples vs. 10,000+ for training from scratch

**Goal 3: Deploy Scalable Computer Vision Infrastructure**
- Real-time wafer map generation from STDF files: convert (lot, wafer, x, y, bin) → 300x300 RGB images
- GPU-accelerated inference: <200ms per wafer map on NVIDIA A10/A100 GPUs
- ONNX model export for cross-platform deployment: PyTorch training → ONNX → TensorRT optimization
- Batch processing: 1,000 wafers/hour sustained throughput
- Multi-resolution support: device-level (300x300), die-level patches (32x32), full-wafer (1024x1024)

**Goal 4: Enable Explainable Predictions**
- Grad-CAM heatmap generation: visualize which wafer regions drive yield predictions
- Attention map overlay: highlight critical spatial zones (e.g., edge dies, center quadrant)
- Feature importance: quantify contribution of spatial patterns vs. parametric features
- Similarity search: find historical wafer maps with similar spatial signatures (cosine similarity >0.85)
- Counterfactual explanations: "If edge dies were passing, predicted yield would increase by 12%"

**Goal 5: Establish ML Best Practices for Computer Vision**
- Automated data versioning: DVC for wafer map datasets (train/val/test splits preserved)
- Experiment tracking: MLflow logging for all hyperparameters, metrics, and model artifacts
- Continuous validation: holdout test sets per product family, temporal validation (train on Q1-Q3, validate on Q4)
- A/B testing framework: compare transfer learning vs. from-scratch models in production
- Model monitoring: track prediction accuracy, confidence calibration, data drift, concept drift

### 3.2 Business Objectives

**Objective 1: Test Cost Reduction**
- **Target**: $3-5M annual savings through 30-40% test time reduction
- **Mechanism**: Early yield prediction → adaptive test termination for high-confidence cases
- **Metric**: Average test time per device (target: 6-8 min vs. baseline 10-15 min)
- **Timeline**: Achieve 50% of target savings within 9 months, 100% within 18 months
- **ROI**: 8-12 month payback period (project cost ~$800K vs. $3-5M annual savings)

**Objective 2: Yield Improvement via Spatial Understanding**
- **Target**: 2-5% yield increase through optimized spatial-aware test limits
- **Mechanism**: Understand edge effect tolerance → relax limits for center dies → reduce false rejects
- **Metric**: Final test yield percentage (target: +2-5 percentage points)
- **Timeline**: Pilot on 1 product (months 6-9), expand to 5 products (months 12-18)
- **Value**: 3% yield increase on TC42x = 15,000 additional good dies/quarter = $1.2M revenue

**Objective 3: NPI Acceleration**
- **Target**: 6-month reduction in new product yield ramp-up time
- **Mechanism**: Transfer learning from mature products → accurate predictions with <500 samples
- **Metric**: Time from first silicon to production release (target: 6 months vs. 12 months baseline)
- **Timeline**: Demonstrate on 1 NPI by month 12, standard practice by month 18
- **Value**: 6-month time-to-market advantage = $8-12M revenue opportunity (automotive design wins)

**Objective 4: Engineering Productivity**
- **Target**: Eliminate 100+ manual wafer map review hours/month
- **Mechanism**: Automated spatial defect classification → engineer reviews only flagged anomalies
- **Metric**: Yield engineer time spent on manual analysis (target: <10 hours/month vs. 40+ baseline)
- **Timeline**: Achieve 50% reduction within 6 months, 80% within 12 months
- **Value**: 3 engineers × 30 hours/month × $80/hour = $86K annual labor savings + redeploy to strategic projects

**Objective 5: Quality and Reliability Improvement**
- **Target**: 30% reduction in DPPM field returns with spatial failure signatures
- **Mechanism**: Identify marginal spatial zones (edge, high-power regions) → tighten screening
- **Metric**: DPPM trend over 12 months (target: 50 PPM → 35 PPM)
- **Timeline**: Baseline current DPPM by spatial pattern (month 3), improvements visible by month 9
- **Value**: 15 PPM reduction × 500K shipped units/year × $200 cost per return = $1.5M annual savings

**Estimated Costs** (12-month project):
- **Personnel**: 4-5 FTE × 12 months × $150K fully loaded = $600K-$750K
- **Infrastructure**: GPU servers (4× A10) + storage = $80K capex + $15K/month opex = $260K
- **Software Licenses**: MLflow, cloud storage, annotation tools = $30K
- **Contingency**: 15% = $130K
- **Total**: ~$1.0M-$1.2M investment for $5M+ annual return = 2.4-4.8× ROI in year 1

### 3.3 Success Metrics

**ML Model Performance Metrics**:
- Yield prediction accuracy: >92% on holdout test sets (per product family)
- Early prediction accuracy: >88% with only 5-10% test completion
- Defect pattern classification: >90% accuracy across 8 defect types (macro F1-score)
- Confidence calibration: Expected Calibration Error (ECE) <5%
- Transfer learning efficiency: Achieve >85% accuracy with 500 samples vs. 10,000 for from-scratch
- Inference latency: <200ms per wafer map (p95) on GPU, <800ms on CPU
- Model size: <100MB ONNX model for edge deployment

**Business Impact Metrics**:
- Test time reduction: 30-40% average across product portfolio (measured monthly)
- Cost savings: $3-5M annual reduction in test costs (tracked quarterly)
- Yield improvement: +2-5 percentage points on pilot products (measured per lot)
- NPI time reduction: 6-month faster yield ramp for new products (measured per NPI)
- DPPM reduction: -30% field returns with spatial signatures (tracked quarterly)
- Engineering productivity: -80% manual wafer map review time (surveyed monthly)

**System Performance Metrics**:
- Data pipeline throughput: Process 1,000 wafers/hour (sustained)
- API response time: <500ms p95 for prediction requests
- System uptime: >99.5% monthly availability
- GPU utilization: 60-80% average (efficient resource usage)
- Storage efficiency: <100MB per wafer map with lossless compression

**Adoption and Usage Metrics**:
- Active users: 20+ yield/test engineers using platform weekly (by month 6)
- Prediction volume: 50,000+ wafer map predictions/month (by month 9)
- Model retraining frequency: Monthly automated retraining with data drift monitoring
- A/B test win rate: Transfer learning models outperform from-scratch models in >80% of tests
- User satisfaction: >4.2/5.0 average rating on quarterly surveys

---

## 4. Target Users/Audience

### 4.1 Primary Users

**Yield Engineers** (Wafer Sort, Final Test, System Level Test)
- Use wafer map predictions to identify systematic spatial failures
- Analyze spatial defect trends across lots, wafers, products
- Investigate yield excursions with spatial root cause analysis
- Optimize test limits based on spatial failure patterns
- Report yield performance with spatial breakdowns to management

**Test Engineers** (ATE Program Owners)
- Configure adaptive test termination based on early yield predictions
- Review prediction confidence distributions to tune termination thresholds
- Debug test program issues flagged by spatial anomaly detection
- Optimize test sequencing using spatial failure timing analysis
- Validate new test insertions against historical spatial patterns

**Product Engineers** (Post-Silicon Validation)
- Monitor spatial yield trends during product ramp (PR7 → PR8 → PR9 → Production)
- Compare spatial signatures across package variants (BGA436 vs. BGA292)
- Identify design-related spatial failures (power droop in center, IR drop at edges)
- Provide spatial failure feedback to design teams for next-gen products
- Validate process changes impact on spatial yield using before/after comparisons

**Manufacturing Engineers** (Production Operations)
- Monitor real-time yield predictions on production floor dashboards
- Respond to spatial excursion alerts within SLA (<4 hours)
- Coordinate containment actions for spatial anomalies (lot holds, re-screen)
- Track adaptive test time savings vs. capacity utilization metrics
- Report production metrics with spatial defect breakdowns

### 4.2 Secondary Users

**Failure Analysis Engineers**
- Prioritize FA samples based on spatial failure predictions (edge vs. center dies)
- Correlate FA findings (delamination, voids, cracks) with predicted spatial patterns
- Validate model predictions through physical failure analysis
- Provide feedback loop: FA results → improve model training labels

**Reliability Engineers**
- Identify marginal spatial zones for targeted reliability screening (HTOL, HAST)
- Correlate spatial failures with stress test results
- Predict field failure risk based on spatial margin analysis
- Design burn-in strategies targeting predicted weak spatial zones

**Quality Engineers**
- Monitor DPPM trends by spatial defect type
- Track customer returns with spatial failure signatures
- Implement spatial-aware outgoing quality control (OQC) strategies
- Report quality metrics with spatial root cause attribution

**Data Scientists / ML Engineers**
- Develop and improve transfer learning models
- Experiment with new architectures (EfficientNet, Vision Transformers)
- Implement domain adaptation techniques
- Monitor model performance and retrain on data drift

**Management / Directors**
- Review executive dashboards: yield trends, cost savings, NPI progress
- Track ROI metrics: test time reduction, DPPM improvement
- Make investment decisions on GPU infrastructure, headcount
- Report AI/ML capabilities to customers and executives

### 4.3 User Personas

**Persona 1: Sarah Chen, Senior Yield Engineer**
- **Background**: 10 years semiconductor yield engineering, specializes in automotive MCUs, expert in wafer map analysis, STDF data mining, SPC
- **Role**: Owns final test yield for TC42x product family across 3 package variants
- **Goals**: Achieve >90% yield, reduce yield excursions by 50%, identify spatial root causes within 24 hours
- **Pain Points**:
  - Manually reviews 200+ wafer maps/week (8 hours/week)
  - Inconsistent spatial pattern identification (edge effect vs. peripheral failure terminology)
  - Delayed excursion detection (2-3 days after lot completion)
  - No quantitative similarity search for historical spatial patterns
- **Technical Skills**: Expert in semiconductor testing, intermediate Python (data analysis), basic ML (understands concepts)
- **Usage Pattern**: Checks wafer map predictions daily, investigates flagged excursions, generates weekly spatial yield reports
- **Success Criteria**: 80% reduction in manual wafer map review time, 90% agreement with model spatial classifications

**Persona 2: Mike Rodriguez, ATE Test Engineer (Advantest V93000)**
- **Background**: 12 years ATE programming, owns test programs for TC41x/TC42x families, expert in parametric testing, scan, MBIST
- **Role**: Develops and maintains test programs, optimizes test time, implements adaptive test strategies
- **Goals**: Reduce test time by 30%, maintain 100% defect coverage, enable adaptive test termination
- **Pain Points**:
  - Test time optimization is manual trial-and-error (4-6 weeks per product)
  - Difficult to predict which tests are redundant vs. critical
  - No systematic way to enable early abort without risking quality
  - Spatial failures not visible in test program (treats all die positions equally)
- **Technical Skills**: Expert in ATE programming, intermediate scripting, basic ML (needs intuitive UI)
- **Usage Pattern**: Reviews early prediction accuracy weekly, configures termination thresholds per product, validates test time savings
- **Success Criteria**: 30% test time reduction achieved, <1% DPPM increase from adaptive termination, high confidence in early predictions

**Persona 3: Dr. Lisa Patel, Product Engineer (Post-Silicon Lead)**
- **Background**: PhD in Electrical Engineering, 8 years post-silicon validation, owns TC43x product family NPI, design-test interface expert
- **Role**: Leads silicon bring-up, characterization, production release for new products
- **Goals**: Achieve 6-month PR7→Production cycle, transfer knowledge from TC42x to TC43x, provide spatial failure feedback to design
- **Pain Points**:
  - 12-month yield ramp is too slow (competitive disadvantage)
  - No spatial predictions available during early production (PR7-PR8, only 100-500 samples)
  - Cannot identify design-related spatial issues early (power grid, IR drop)
  - Knowledge from mature products (TC42x) not leveraged for new products (TC43x)
- **Technical Skills**: Expert in semiconductor physics and design, advanced data analysis, intermediate ML (familiar with transfer learning concepts)
- **Usage Pattern**: Uses transfer learning models for NPI, compares spatial signatures across products/milestones, provides feedback to design teams
- **Success Criteria**: 6-month NPI cycle achieved, spatial predictions available with <500 samples, design feedback loop established

**Persona 4: James Kim, Manufacturing Operations Engineer**
- **Background**: 15 years semiconductor manufacturing, real-time production floor management, MES integration expert
- **Role**: Ensures production targets met, responds to excursions, coordinates operations across test/FA/quality
- **Goals**: Maximize tester utilization (>90%), minimize WIP, respond to excursions within 4 hours
- **Pain Points**:
  - Yield excursions discovered after lot completion (too late for containment)
  - Tester capacity constraints during peak demand (2-week backlog)
  - No real-time spatial yield visibility (waits for offline analysis)
  - Manual escalation processes (email/phone calls)
- **Technical Skills**: Expert in manufacturing operations, intermediate data analysis, basic ML (dashboard user)
- **Usage Pattern**: Monitors real-time yield dashboard hourly, responds to spatial excursion alerts, tracks test time savings impact on capacity
- **Success Criteria**: <4 hour excursion response time, 30% capacity increase from test time reduction, automated alert workflows

---

## 5. User Stories

**US-001: Early Yield Prediction for Adaptive Test Termination**
- **As a** Test Engineer
- **I want to** predict final yield after 5-10% test completion with >88% accuracy
- **So that** I can enable adaptive test termination for high-confidence predictions, reducing test time by 30-40%
- **Acceptance Criteria**:
  - Given a wafer with 500 tests completed (of 10,000 total)
  - When the model processes wafer map and partial test data
  - Then yield prediction is provided with confidence score (0-100%)
  - And prediction accuracy >88% validated on holdout test set
  - And predictions available within 200ms (real-time inference)
  - And model can flag low-confidence cases for full test completion

**US-002: Automated Spatial Defect Classification**
- **As a** Yield Engineer
- **I want to** automatically classify wafer maps into 8 defect types (edge, center, ring, quadrant, scratch, random, mixed, normal)
- **So that** I can eliminate 100+ manual review hours/month and ensure consistent spatial pattern identification
- **Acceptance Criteria**:
  - Given a wafer map image (300x300 RGB)
  - When the ResNet model processes the image
  - Then defect type classification is provided with confidence scores for all 8 types
  - And classification accuracy >90% (macro F1-score) on labeled validation set
  - And classifications match expert yield engineer labels in >85% of cases
  - And batch processing supports 1,000 wafers/hour throughput

**US-003: Transfer Learning for New Product Rapid Adaptation**
- **As a** Product Engineer
- **I want to** achieve >85% yield prediction accuracy with only 500 training samples for new products
- **So that** I can enable yield predictions during early NPI (PR7-PR8) instead of waiting 6-12 months for data collection
- **Acceptance Criteria**:
  - Given a new product (TC43x) with 500 labeled wafer maps
  - When transfer learning is applied from mature product (TC42x) models
  - Then yield prediction accuracy >85% on new product validation set
  - And accuracy comparable to from-scratch model trained on 10,000 samples
  - And fine-tuning completes within 4 hours on GPU
  - And model ready for production deployment after validation

**US-004: Spatial Root Cause Visualization with Grad-CAM**
- **As a** Yield Engineer
- **I want to** see Grad-CAM heatmaps highlighting which wafer regions drive yield predictions
- **So that** I can quickly identify spatial root causes (edge effect, center cluster) without manual analysis
- **Acceptance Criteria**:
  - Given a wafer map with low yield prediction (e.g., 65%)
  - When Grad-CAM visualization is requested
  - Then heatmap overlay shows attention weights on wafer map (high attention = red, low = blue)
  - And highlighted regions correlate with actual failure zones (validated by expert review)
  - And visualization generated within 500ms
  - And engineer can export heatmap for reports/presentations

**US-005: Multi-Product Model Comparison and Selection**
- **As a** Data Scientist
- **I want to** compare multiple transfer learning strategies (freeze-all vs. progressive unfreezing vs. full fine-tuning) via A/B testing
- **So that** I can select the optimal approach per product family based on accuracy, training time, and sample efficiency
- **Acceptance Criteria**:
  - Given 3 transfer learning strategies implemented
  - When models are trained on same dataset splits
  - Then MLflow tracks all experiments with hyperparameters, metrics, artifacts
  - And A/B test framework compares strategies on production traffic (shadow mode)
  - And statistical significance tests determine winning strategy (p<0.05)
  - And winning model promoted to production automatically

**US-006: Real-Time Spatial Excursion Detection**
- **As a** Manufacturing Operations Engineer
- **I want to** receive alerts when wafer-level yield predictions drop below threshold (e.g., <75%) indicating spatial excursion
- **So that** I can respond within 4 hours (vs. 48 hours currently) and minimize scrap exposure
- **Acceptance Criteria**:
  - Given real-time wafer map generation from STDF streaming
  - When yield prediction for a wafer falls below 75%
  - Then alert is triggered via email, Slack, and MES integration
  - And alert includes wafer ID, predicted yield, confidence, defect type classification
  - And historical comparison shows similar past excursions
  - And alert latency <15 minutes from test completion

**US-007: Cross-Package Transfer Learning Validation**
- **As a** Product Engineer
- **I want to** validate that models trained on BGA436 package transfer to BGA292 package (same die, different package)
- **So that** I can avoid retraining from scratch for each package variant, saving 6 weeks per variant
- **Acceptance Criteria**:
  - Given BGA436 trained model (10,000 samples)
  - When fine-tuned on BGA292 with 1,000 samples
  - Then BGA292 accuracy >88% (vs. >92% for BGA436)
  - And fine-tuning completes within 2 hours
  - And spatial defect patterns transfer (edge effect detection consistent across packages)
  - And model ready for production after validation

**US-008: Confidence-Based Test Strategy Routing**
- **As a** Test Engineer
- **I want to** route devices to different test strategies based on prediction confidence: high-confidence PASS → early termination, low-confidence or FAIL → extended characterization
- **So that** I can optimize test resources (fast test for clear cases, deep analysis for suspect cases)
- **Acceptance Criteria**:
  - Given yield prediction with confidence score
  - When confidence >95% and predicted yield >85% → early termination
  - When confidence <80% or predicted yield <70% → full test + extended characterization
  - Then test time savings >30% on average across all devices
  - And DPPM increase <1% (quality not compromised)
  - And tester capacity increase >25% (effective throughput)

---

## 6. Functional Requirements

### 6.1 Core Features

**FR-001: STDF Data Ingestion and Wafer Map Generation**
- Ingest STDF files from Advantest V93000 and Teradyne testers with >99% parse success rate
- Extract device coordinates (lot_id, wafer_id, x, y), bin assignments, test results
- Generate wafer map images (300x300 RGB) with color-coded bins: PASS (green), FAIL (red/orange/yellow by bin), NOTEST (gray)
- Support retest data merging: combine first run + rerun results (use latest valid result per die)
- Handle multiple test stages: Wafer Sort (WS), Final Test (FT), System Level Test (SLT)
- Normalize die coordinates across different wafer sizes (200mm, 300mm) and orientations (flat, notch)

**FR-002: ResNet Transfer Learning Model Training**
- Support ResNet-18 (11M parameters) and ResNet-50 (25M parameters) architectures with ImageNet pre-trained weights
- Implement progressive fine-tuning strategy:
  - Phase 1: Freeze all layers except final FC layer (classifier head), train for 1-2 epochs
  - Phase 2: Unfreeze last ResNet block (layer4), train for 2-3 epochs with lower learning rate
  - Phase 3: Fine-tune entire network with very low learning rate for 5-10 epochs
- Support discriminative learning rates: 1e-3 for classifier, 1e-4 for last block, 1e-5 for earlier layers
- Implement domain adaptation: batch normalization statistics adaptation from ImageNet to wafer maps
- Enable multi-GPU training with DistributedDataParallel for datasets >50,000 samples

**FR-003: Data Augmentation Pipeline**
- Geometric augmentations: rotations (0°, 90°, 180°, 270°), horizontal/vertical flips, random crops
- Color augmentations: brightness (±20%), contrast (±20%), saturation (±10%), hue (±5%)
- Spatial augmentations: random elastic transforms, grid distortions (simulate wafer warping)
- Test-time augmentation (TTA): apply 8 augmentations during inference, average predictions for robustness
- Augmentation probability control: apply each augmentation with p=0.5 probability during training
- Preserve wafer structure: rotations must be 90° multiples (wafer symmetry), no arbitrary rotations

**FR-004: Multi-Class Yield and Defect Classification**
- Yield regression: predict continuous yield percentage (0-100%) with Mean Absolute Error (MAE) <3%
- Defect classification: 8-class classifier (Edge Effect, Center Cluster, Ring Pattern, Quadrant Failure, Scratch/Line, Random Failure, Mixed-Mode, Normal)
- Multi-task learning: joint optimization of yield regression + defect classification with weighted loss
- Confidence scoring: output calibrated confidence scores via temperature scaling and Monte Carlo Dropout (20 forward passes)
- Per-class confidence: provide confidence scores for each of 8 defect types (support multi-label scenarios)

**FR-005: Early Prediction with Partial Test Data**
- Generate wafer maps from partial test results: 5%, 10%, 20%, 50% test completion milestones
- Train separate models for each completion milestone or unified model with completion % as input feature
- Achieve accuracy targets: 5% completion (>80%), 10% completion (>88%), 20% completion (>92%)
- Confidence calibration: lower confidence scores for earlier predictions (reflect uncertainty)
- Support adaptive inference: if early prediction has high confidence, skip recomputation at later milestones

**FR-006: Grad-CAM Explainability and Attention Visualization**
- Generate Grad-CAM heatmaps: visualize which wafer regions contribute most to yield predictions
- Overlay heatmaps on original wafer maps with transparency control (alpha=0.4)
- Support multiple Grad-CAM layers: layer4 (high-level features), layer3 (mid-level), layer2 (low-level)
- Export heatmaps as PNG images for reports and presentations
- Batch Grad-CAM: generate heatmaps for all wafers in a lot for comparative analysis

**FR-007: GPU-Accelerated Inference with ONNX Export**
- Export trained PyTorch models to ONNX format for cross-platform deployment
- Optimize ONNX models with TensorRT for NVIDIA GPUs (FP16 precision, kernel fusion)
- Support CPU inference with ONNX Runtime for edge deployment (lower latency than PyTorch)
- Batch inference: process 64 wafer maps in parallel on GPU (<200ms total latency)
- Dynamic batching: automatically batch multiple API requests for efficiency

**FR-008: Experiment Tracking and Model Versioning**
- Log all training runs to MLflow: hyperparameters, metrics (accuracy, loss, F1), artifacts (model checkpoints, Grad-CAM samples)
- Track data provenance: training dataset versions (DVC), STDF file sources, preprocessing parameters
- Model registry: register production models with version tags (v1.0, v1.1), stage (staging, production)
- Lineage tracking: trace predictions back to model version, training data, hyperparameters
- Compare experiments: side-by-side comparison of accuracy, training time, inference latency

**FR-009: Automated Model Retraining Pipeline**
- Schedule retraining: monthly or triggered by data drift detection (>10% accuracy drop)
- Incremental training: fine-tune existing model on new data vs. full retraining from scratch
- Validation gating: new model promoted to production only if accuracy improvement >2% on holdout set
- Rollback mechanism: revert to previous model version if production accuracy degrades
- A/B testing: shadow mode validation comparing old vs. new model for 1 week before full rollout

**FR-010: Multi-Resolution Wafer Map Analysis**
- Device-level: 300x300 images showing all die on wafer (standard resolution)
- Die-level patches: 32x32 patches extracted per die for fine-grained analysis
- Full-wafer: 1024x1024 high-resolution images for detailed spatial pattern analysis
- Hierarchical predictions: aggregate die-level predictions to wafer-level yield forecast
- Zoom capability: UI allows zooming into specific wafer regions for detailed inspection

**FR-011: Historical Similarity Search**
- Embed wafer maps into 512-dimensional feature vectors using ResNet backbone (before final FC layer)
- Index embeddings in vector database (FAISS, Annoy, or Milvus) for fast nearest-neighbor search
- Query: "Find wafer maps similar to current wafer" → return top-10 with cosine similarity >0.85
- Display results: show similar historical wafers with metadata (lot, date, yield, defect type)
- Use cases: "Has this spatial pattern occurred before?", "What was the root cause last time?"

**FR-012: Multi-Product Model Management**
- Support separate models per product family: TC41x, TC42x, TC43x
- Cross-product transfer: initialize TC43x model from TC42x weights (related products)
- Model selection API: automatically route inference requests to correct model based on product ID
- Unified model option: train single model on all products with product ID as input feature
- Performance comparison: track accuracy per product, identify products needing specialized models

### 6.2 Advanced Features

**FR-013: Adversarial Domain Adaptation**
- Implement domain adversarial training: shared feature extractor + yield predictor + domain discriminator
- Goal: learn features invariant across ImageNet and semiconductor domains
- Loss function: yield prediction loss - λ × domain classification loss (gradient reversal)
- Evaluation: measure domain shift via Maximum Mean Discrepancy (MMD) before/after adaptation
- Optional feature: enable only if standard transfer learning accuracy <88%

**FR-014: Multi-Modal Fusion (Wafer Maps + Parametric Data)**
- Combine wafer map image features (ResNet) with parametric test statistics (IDDQ mean/std, Vth median, Fmax p5/p95)
- Fusion architecture: concatenate image embeddings (512-dim) + parametric features (50-dim) → final FC layers
- Hypothesis: spatial + parametric information improves accuracy by 3-5% vs. wafer maps alone
- Validation: A/B test on production data to confirm multi-modal benefit

**FR-015: Active Learning for Efficient Labeling**
- Identify wafer maps with high prediction uncertainty (low confidence or high variance in MC Dropout)
- Present uncertain cases to yield engineers for manual labeling (spatial defect type, root cause)
- Prioritize labeling: focus on edge cases, rare defect patterns, new products
- Retrain model on newly labeled data quarterly
- Track labeling efficiency: achieve 95% accuracy with 30% less labeled data vs. random sampling

**FR-016: Temporal Yield Trend Analysis**
- Track wafer map predictions over time: daily, weekly, monthly yield trends per product
- Detect concept drift: yield prediction accuracy degrading over time (model needs retraining)
- Seasonal analysis: compare yield patterns across months (thermal effects, process variations)
- Lot-to-lot correlation: identify systematic trends across consecutive lots (process drift)
- Visualization: time series charts of predicted yield, defect type distribution over time

**FR-017: Counterfactual Explanation Generation**
- Generate counterfactual wafer maps: "What if edge dies were passing? Predicted yield would increase by 12%"
- Method: perturb wafer map by flipping edge die bins from FAIL→PASS, recompute prediction
- Use cases: quantify impact of spatial defects, prioritize FA efforts, estimate yield recovery potential
- Visualization: side-by-side comparison of original vs. counterfactual wafer maps with yield delta

**FR-018: Cross-Fab and Cross-Package Transfer Learning**
- Transfer models trained on Fab A data to Fab B (different foundry, same design)
- Transfer models from BGA436 package to BGA292 (same die, different package)
- Measure transfer gap: accuracy drop when applying model to new domain (target <10%)
- Fine-tuning strategy: adapt model with 500-1,000 samples from target domain
- Validation: ensure spatial defect patterns transfer correctly (edge effect detection consistent)

---

## 7. Non-Functional Requirements

### 7.1 Performance

**Inference Latency**:
- Single wafer map prediction: <200ms p95 on GPU (NVIDIA A10/A100), <800ms p95 on CPU
- Batch inference (64 wafer maps): <5 seconds total on GPU (<80ms per wafer amortized)
- Grad-CAM heatmap generation: <500ms additional latency on GPU
- Real-time streaming: process 1 wafer/second sustained throughput for production floor monitoring
- Early prediction (5-10% completion): <150ms latency (faster due to smaller input data)

**Model Training Performance**:
- Phase 1 (freeze backbone): 1-2 epochs, 2-4 hours on single A10 GPU (10,000 samples)
- Phase 2 (unfreeze last block): 2-3 epochs, 4-8 hours on single A10 GPU
- Phase 3 (full fine-tuning): 5-10 epochs, 10-20 hours on single A10 GPU
- Transfer learning speedup: 5-10× faster than training from scratch (100+ hours)
- Multi-GPU scaling: 2× speedup with 2 GPUs, 3.5× with 4 GPUs (diminishing returns due to batch size limits)

**Data Pipeline Performance**:
- STDF parsing to wafer map generation: <30 seconds per wafer (5,000 dies)
- Wafer map image rendering: <5 seconds per 300x300 image
- Batch wafer map generation: 1,000 wafers/hour sustained
- Data augmentation: <100ms per image (on-the-fly during training)

**Model Size and Memory**:
- ResNet-18: 44MB model size, 2GB GPU memory for inference (batch=32)
- ResNet-50: 98MB model size, 4GB GPU memory for inference (batch=32)
- ONNX optimized: 30-40% size reduction with INT8 quantization
- Training memory: 8-16GB GPU for ResNet-18, 16-24GB for ResNet-50

### 7.2 Reliability

**System Availability**:
- Uptime SLA: 99.5% monthly (max 3.6 hours downtime/month)
- Inference service availability: 99.9% (critical path for production)
- Training pipeline availability: 99.0% (scheduled maintenance windows acceptable)
- GPU failover: automatic fallback to CPU inference if GPU unavailable (<5 second switchover)

**Data Durability and Integrity**:
- Wafer map images: 99.999999999% durability (11 nines) on S3/MinIO object storage
- Model checkpoints: versioned with DVC, stored in redundant storage (3 replicas minimum)
- Training data: checksums validated on ingestion, corrupted files quarantined
- Exactly-once semantics: idempotent wafer map generation (same STDF → same wafer map)

**Model Reliability**:
- Prediction stability: same wafer map → same prediction (deterministic inference)
- Confidence calibration: Expected Calibration Error (ECE) <5% (predictions match actual accuracy)
- Graceful degradation: if model unavailable, fallback to rule-based spatial analysis
- Version rollback: automatic rollback if new model accuracy drops >3% in production

**Error Handling and Recovery**:
- Failed inference requests: retry with exponential backoff (3 attempts, max 10 second delay)
- Out-of-memory errors: automatic batch size reduction and retry
- Corrupted model files: checksum validation before loading, fallback to previous version
- GPU errors: automatic CPU fallback, alert operations team

**Disaster Recovery**:
- Recovery Point Objective (RPO): <4 hours (max data loss)
- Recovery Time Objective (RTO): <2 hours (max downtime for full system restore)
- Backup frequency: daily model checkpoints, hourly wafer map backups
- Multi-region replication: critical models replicated to secondary region

### 7.3 Usability

**User Interface**:
- Intuitive wafer map visualization: pan, zoom, hover for die details
- One-click prediction: upload STDF → get yield forecast in 3 clicks
- Grad-CAM overlay: toggle on/off heatmap visualization
- Responsive design: works on desktop (primary), tablet, mobile (view-only)
- Dashboard load time: <2 seconds for initial page load, <500ms for interactions

**Learning Curve**:
- New user onboarding: <2 hours with guided tutorial and sample datasets
- Basic tasks (upload STDF, view prediction): trainable in <30 minutes
- Advanced tasks (model retraining, hyperparameter tuning): trainable in <4 hours for ML-familiar users
- Documentation: inline help, video tutorials, API docs, troubleshooting guides

**Accessibility**:
- WCAG 2.1 AA compliance: keyboard navigation, screen reader support, color contrast ratios
- Colorblind-friendly palettes: wafer map color schemes accessible to deuteranopia, protanopia
- Multi-language support: English (primary), additional languages if needed (German, Chinese for global teams)
- Keyboard shortcuts: common actions (refresh, zoom, export) accessible via hotkeys

**User Feedback and Notifications**:
- Real-time progress indicators: wafer map generation, model inference, training progress bars
- Clear error messages: actionable guidance for user errors (e.g., "STDF file corrupted at record 1234, re-upload")
- Success confirmations: toast notifications for completed actions
- Email/Slack alerts: configurable notifications for model training completion, accuracy degradation

### 7.4 Maintainability

**Code Quality**:
- Modular architecture: separate modules for data ingestion, model training, inference, visualization
- Type hints: Python 3.11+ type annotations for all functions and classes
- Code coverage: >90% unit test coverage for core modules, >70% for integration tests
- Linting and formatting: Black, isort, flake8, mypy enforced in CI/CD pipeline

**Documentation**:
- Code documentation: docstrings for all public APIs (Google style)
- Architecture docs: system diagrams, data flow diagrams, API specifications
- Runbooks: operational procedures for common tasks (model deployment, rollback, debugging)
- Change logs: detailed release notes for each version

**Dependency Management**:
- Pinned dependencies: requirements.txt with exact versions (e.g., torch==2.4.0, not torch>=2.4)
- Dependency vulnerability scanning: Dependabot, Snyk for CVE detection
- Minimal dependencies: avoid unnecessary libraries to reduce attack surface and maintenance burden
- License compliance: all dependencies use permissive licenses (MIT, Apache 2.0, BSD)

**Monitoring and Observability**:
- Structured logging: JSON logs with correlation IDs, user IDs, request IDs
- Distributed tracing: OpenTelemetry for end-to-end request tracing
- Metrics instrumentation: Prometheus metrics for latency, throughput, error rates, GPU utilization
- Centralized log aggregation: OpenSearch/ELK stack for log search and analysis

**Deployment Automation**:
- Infrastructure as Code (IaC): Terraform for cloud resources, Helm for Kubernetes deployments
- GitOps: ArgoCD for declarative deployment management
- Automated rollback: canary deployments with automatic rollback on error rate >5%
- Blue-green deployments: zero-downtime model updates

---

## 8. Technical Requirements

### 8.1 Technical Stack

**Core ML Framework**:
- **PyTorch 2.4+**: Primary deep learning framework for model development and training
- **torchvision 0.19+**: Pre-trained ResNet models, image transforms, data loaders
- **PyTorch Lightning 2.2+**: High-level training framework for multi-GPU, mixed precision, distributed training
- **ONNX 1.16+**: Model export format for cross-platform deployment and optimization
- **TensorRT 8.6+**: NVIDIA GPU inference optimization (FP16/INT8 quantization, kernel fusion)

**Transfer Learning & Computer Vision**:
- **ResNet-18/ResNet-50**: Pre-trained on ImageNet (1.2M images, 1000 classes), adapted for wafer maps
- **torchvision.models**: Access to ImageNet weights (IMAGENET1K_V1, IMAGENET1K_V2 for improved accuracy)
- **albumentations 1.4+**: Advanced image augmentation library (elastic transforms, grid distortions, CLAHE)
- **OpenCV 4.10+**: Image processing, wafer map rendering, spatial transformations
- **PIL/Pillow 10.3+**: Image I/O, format conversions, thumbnail generation

**Domain Adaptation Techniques**:
- **Batch Normalization Adaptation**: Update BN statistics on semiconductor data (running mean/variance)
- **Discriminative Learning Rates**: Separate LR schedules per layer group (AdamW with param_groups)
- **Feature Alignment (optional)**: Maximum Mean Discrepancy (MMD) loss for domain invariance
- **Adversarial Domain Adaptation (optional)**: Gradient reversal layer for domain-invariant features

**Data Augmentation**:
- **Geometric**: Rotations (90°, 180°, 270°), horizontal/vertical flips, random crops (224x224 from 256x256)
- **Color**: Random brightness (±20%), contrast (±20%), saturation (±10%), hue (±5%)
- **Spatial**: Elastic transforms, grid distortions (simulate wafer warping), affine transforms
- **Test-Time Augmentation (TTA)**: 8 augmentations during inference, average predictions for robustness

**Model Optimization**:
- **Mixed Precision Training**: Automatic Mixed Precision (AMP) with torch.cuda.amp for 2-3× speedup
- **Gradient Accumulation**: Simulate larger batch sizes on limited GPU memory (accumulate 4-8 steps)
- **Model Pruning**: Magnitude-based pruning for 30-40% parameter reduction with <2% accuracy loss
- **Knowledge Distillation**: Distill ResNet-50 → ResNet-18 for faster inference with minimal accuracy trade-off

**Backend Services**:
- **Python 3.11+**: Latest stable Python with performance improvements (10-25% faster than 3.9)
- **FastAPI 0.115+**: Modern async web framework for REST API endpoints
- **Uvicorn 0.30+**: ASGI server with HTTP/2 support, WebSocket for real-time updates
- **Pydantic 2.8+**: Data validation and serialization with type safety

**Data Processing**:
- **Pandas 2.2+**: STDF data manipulation, wafer-level aggregations
- **NumPy 1.26+**: Numerical computations, wafer map array operations
- **Parquet via PyArrow 16.0+**: Columnar storage for fast I/O and compression
- **DuckDB 1.0+**: In-process SQL analytics for wafer map queries

**Experiment Tracking & MLOps**:
- **MLflow 2.12+**: Experiment tracking, model registry, model versioning
- **DVC 3.50+**: Data version control for wafer map datasets, model lineage
- **Weights & Biases (W&B) 0.17+**: (optional) Advanced experiment tracking with visualizations
- **Git LFS 3.5+**: Large file storage for model checkpoints (<100MB limit per file in Git)

**Visualization**:
- **Matplotlib 3.8+**: Wafer map plotting, training curves, confusion matrices
- **Plotly 5.20+**: Interactive dashboards, 3D wafer visualizations, Grad-CAM overlays
- **TensorBoard 2.16+**: Real-time training monitoring (loss curves, learning rate, gradients)
- **Seaborn 0.13+**: Statistical visualizations, correlation heatmaps

**Deployment & Infrastructure**:
- **Docker 26.0+**: Containerization for reproducible environments
- **Kubernetes 1.30+**: Container orchestration, auto-scaling, load balancing
- **Helm 3.14+**: Kubernetes package manager for application deployment
- **NGINX 1.26+**: Reverse proxy, load balancer, SSL termination
- **Redis 7.2+**: Caching layer for frequent wafer map queries, prediction results

**GPU Acceleration**:
- **CUDA 12.4+**: NVIDIA GPU programming framework
- **cuDNN 9.0+**: GPU-accelerated deep learning primitives
- **NVIDIA A10/A100 GPUs**: Production inference (A10) and training (A100)
- **NVIDIA Docker Runtime**: GPU passthrough to containers

**Storage**:
- **MinIO 2024.5+**: S3-compatible object storage for wafer map images, model checkpoints
- **PostgreSQL 16+**: Relational database for metadata (model versions, experiment configs, user data)
- **FAISS 1.8+ / Milvus 2.4+**: Vector database for wafer map embedding similarity search

**Monitoring & Observability**:
- **Prometheus 2.52+**: Metrics collection (latency, throughput, GPU utilization, error rates)
- **Grafana 11.0+**: Metrics visualization dashboards
- **OpenSearch 2.13+ / ELK Stack**: Log aggregation and full-text search
- **Jaeger 1.56+ / OpenTelemetry 1.24+**: Distributed tracing for request flows

**CI/CD & DevOps**:
- **GitHub Actions / GitLab CI**: Automated testing, linting, Docker builds
- **ArgoCD 2.11+**: GitOps continuous delivery for Kubernetes
- **Terraform 1.8+**: Infrastructure as Code for cloud resources (AWS/Azure/GCP)
- **Ansible 2.16+**: Configuration management for on-prem servers

**Security**:
- **OAuth2/OIDC**: Authentication via Azure AD, Okta, or Auth0
- **JWT**: Stateless authentication tokens
- **HashiCorp Vault 1.16+**: Secrets management (API keys, database passwords, model encryption keys)
- **Trivy 0.51+**: Container vulnerability scanning
- **SAST/DAST**: Static/dynamic application security testing (Snyk, SonarQube)

### 8.2 AI/ML Components

**ResNet Architecture Details**:
- **ResNet-18**: 11.7M parameters, 18 layers (8 residual blocks), 1.8 GFLOPs for 224x224 input
  - Layer structure: Conv1 (7x7, 64) → MaxPool → Layer1 (2×BasicBlock, 64) → Layer2 (2×BasicBlock, 128) → Layer3 (2×BasicBlock, 256) → Layer4 (2×BasicBlock, 512) → AvgPool → FC
  - Inference time: ~5ms on A10 GPU, ~50ms on CPU (single image)
  - Suitable for: rapid prototyping, edge deployment, real-time inference
- **ResNet-50**: 25.6M parameters, 50 layers (16 residual blocks), 4.1 GFLOPs for 224x224 input
  - Layer structure: Conv1 (7x7, 64) → MaxPool → Layer1 (3×Bottleneck, 256) → Layer2 (4×Bottleneck, 512) → Layer3 (6×Bottleneck, 1024) → Layer4 (3×Bottleneck, 2048) → AvgPool → FC
  - Inference time: ~12ms on A10 GPU, ~120ms on CPU (single image)
  - Suitable for: higher accuracy requirements, offline batch processing

**ImageNet Pre-trained Weights**:
- **Source**: torchvision.models.resnet18(weights='IMAGENET1K_V1') or 'IMAGENET1K_V2' (improved accuracy)
- **Training**: 1.28M images from 1000 classes, trained for 90 epochs with SGD + momentum
- **ImageNet Performance**: ResNet-18 (69.8% top-1, 89.1% top-5), ResNet-50 (76.1% top-1, 92.9% top-5)
- **Transfer Hypothesis**: Low-level features (edges, textures in early layers) transfer well to wafer maps; high-level features (object parts in later layers) require fine-tuning

**Progressive Fine-Tuning Strategy**:
- **Phase 1: Freeze Backbone, Train Classifier** (1-2 epochs, ~2-4 hours)
  - Freeze all ResNet layers except final FC layer
  - Replace FC layer: in_features=512 (ResNet-18) or 2048 (ResNet-50) → out_features=8 (defect classes) or 1 (yield regression)
  - Optimizer: AdamW (lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999))
  - Goal: Adapt classifier to semiconductor domain while preserving ImageNet features
  - Expected accuracy: 75-82% (better than random, leverages pre-trained features)
  
- **Phase 2: Unfreeze Last Block, Fine-Tune** (2-3 epochs, ~4-8 hours)
  - Unfreeze Layer4 (last residual block): 2 BasicBlocks (ResNet-18) or 3 Bottlenecks (ResNet-50)
  - Discriminative learning rates: Layer4 (lr=1e-4), Classifier (lr=1e-3)
  - Goal: Adapt high-level features to wafer-specific spatial patterns
  - Expected accuracy: 85-90% (significant improvement as Layer4 learns wafer-specific features)
  
- **Phase 3: Full Fine-Tuning** (5-10 epochs, ~10-20 hours)
  - Unfreeze all layers with discriminative learning rates:
    - Layer1-2 (early layers): lr=1e-5 (minimal adaptation, preserve edges/textures)
    - Layer3 (mid layers): lr=5e-5
    - Layer4 (late layers): lr=1e-4
    - Classifier: lr=1e-3
  - Learning rate schedule: CosineAnnealingLR or OneCycleLR with warmup
  - Goal: Fine-tune entire network for optimal wafer map accuracy
  - Expected accuracy: 92-95% (full adaptation to semiconductor domain)

**Domain Adaptation Techniques**:
- **Batch Normalization (BN) Adaptation**:
  - Update BN running_mean and running_var on semiconductor data (wafer maps)
  - Method: model.train() mode during first epoch to recompute BN statistics, then model.eval()
  - Rationale: ImageNet BN statistics (natural images) differ from wafer maps (binary spatial patterns)
  
- **Discriminative Learning Rates**:
  - Lower layers (Layer1-2): small LR (1e-5) → preserve general features (edges, blobs)
  - Higher layers (Layer3-4): larger LR (1e-4) → adapt to wafer-specific patterns (spatial distributions)
  - Classifier head: largest LR (1e-3) → learn from scratch for semiconductor task
  
- **Feature Alignment via Maximum Mean Discrepancy (MMD)** (optional, if standard transfer fails):
  - Measure domain shift between ImageNet and wafer map feature distributions
  - Add MMD loss: minimize difference in mean embeddings between domains
  - Implementation: compute kernel embeddings (RBF kernel) of Layer4 outputs, add to loss function
  
- **Adversarial Domain Adaptation** (optional, advanced):
  - Architecture: Shared feature extractor (ResNet backbone) + Task classifier (yield/defect) + Domain discriminator (ImageNet vs. wafer)
  - Training: Gradient Reversal Layer (GRL) to learn domain-invariant features
  - Loss: L_total = L_task - λ × L_domain (λ=0.1-0.5, tuned via validation)
  - Use case: if >10% accuracy gap between ImageNet and wafer domains

**Data Augmentation Pipeline**:
- **Training Augmentations** (applied with p=0.5 probability each):
  - Resize: 256x256 (upscale from 300x300 raw wafer maps or downscale if larger)
  - RandomResizedCrop: 224x224 (random crop + resize, scale=(0.8, 1.0), ratio=(0.9, 1.1))
  - RandomRotation: ±90° or ±180° (wafer symmetry-preserving, no arbitrary angles)
  - RandomHorizontalFlip: p=0.5
  - RandomVerticalFlip: p=0.5
  - ColorJitter: brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
  - ElasticTransform: α=50, σ=5 (simulate wafer warping, probe card pressure)
  - GridDistortion: num_steps=5, distort_limit=0.3 (simulate wafer non-planarity)
  - ToTensor: convert PIL Image → torch.Tensor (C, H, W)
  - Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] (ImageNet statistics)
  
- **Validation/Test Augmentations** (no randomness):
  - Resize: 224x224 (center crop or resize)
  - ToTensor
  - Normalize: ImageNet statistics
  
- **Test-Time Augmentation (TTA)** (optional, for robust predictions):
  - Apply 8 augmentations: original, H-flip, V-flip, HV-flip, Rot90, Rot90+H-flip, Rot90+V-flip, Rot90+HV-flip
  - Forward pass each augmentation through model, average predictions: y_final = mean([y1, y2, ..., y8])
  - Trade-off: 8× inference time for 1-2% accuracy improvement

**Model Optimization Techniques**:
- **Mixed Precision Training (AMP)**:
  - Forward/backward in FP16 (half precision) for 2-3× speedup, weights in FP32
  - torch.cuda.amp.GradScaler for automatic loss scaling (prevent underflow)
  - Memory savings: ~40% GPU memory reduction
  
- **Gradient Accumulation**:
  - Effective batch size = batch_per_gpu × accumulation_steps × num_gpus
  - Example: 16 (batch_per_gpu) × 4 (accum_steps) × 2 (GPUs) = 128 effective batch
  - Use case: simulate large batches on limited GPU memory
  
- **Model Pruning**:
  - Magnitude-based pruning: remove weights with |weight| < threshold
  - Target: 30-40% sparsity (FLOPs reduction) with <2% accuracy loss
  - Method: torch.nn.utils.prune with iterative pruning schedule
  
- **Knowledge Distillation**:
  - Teacher: ResNet-50 (25.6M params, 92% accuracy)
  - Student: ResNet-18 (11.7M params, target 90% accuracy)
  - Loss: L = α × L_CE(y_student, y_true) + (1-α) × L_KL(y_student, y_teacher), α=0.7
  - Benefit: 2× faster inference with minimal accuracy trade-off

**Loss Functions**:
- **Yield Regression**: Mean Squared Error (MSE) or Smooth L1 Loss (Huber)
  - MSE for well-behaved data, Huber for outliers (scrap lots)
  - Output activation: Sigmoid (0-1 range) × 100 (0-100% yield)
  
- **Defect Classification**: CrossEntropyLoss with class weights
  - Class weights: inverse frequency to handle imbalance (edge effect 40% → weight=1/0.4=2.5)
  - Label smoothing: ε=0.1 to prevent overconfident predictions
  
- **Multi-Task Learning** (yield + defect):
  - L_total = λ_yield × MSE_yield + λ_defect × CE_defect
  - Weights: λ_yield=0.7, λ_defect=0.3 (tuned via validation)

**Evaluation Metrics**:
- **Yield Prediction**: MAE, RMSE, R² score
- **Defect Classification**: Accuracy, Macro F1-score, Per-class recall/precision, Confusion matrix
- **Confidence Calibration**: Expected Calibration Error (ECE), reliability diagrams
- **Transfer Learning Efficiency**: Accuracy vs. training samples curve (50, 100, 500, 1000, 5000, 10000 samples)

---

## 9. System Architecture

### 9.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           STDF Data Sources (ATE Testers)                       │
│                    Advantest V93000 SMT8  |  Teradyne Testers                   │
└────────────────────────────────┬────────────────────────────────────────────────┘
                                 │ STDF Files (FTP/S3)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Data Ingestion & Processing Layer                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌───────────────┐    ┌────────────────┐                  │
│  │ STDF Parser  │ -> │ Wafer Map Gen │ -> │ Preprocessing  │                  │
│  │ (pystdf)     │    │ (300x300 RGB) │    │ (Resize/Norm)  │                  │
│  └──────────────┘    └───────────────┘    └────────────────┘                  │
│         │                     │                     │                           │
│         └─────────────────────┴─────────────────────┘                           │
│                               │                                                 │
│                               ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │           MinIO Object Storage (Wafer Map Images + Metadata)            │   │
│  │              Parquet: test_results, wafer_metadata, lot_info            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ML Training & Inference Layer                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    Training Pipeline (Offline)                          │   │
│  │  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │   │
│  │  │ DataLoader   │->│ ResNet-18/50│->│ Progressive  │->│ MLflow     │  │   │
│  │  │ (Augment)    │  │ (PyTorch)   │  │ Fine-Tuning  │  │ Tracking   │  │   │
│  │  └──────────────┘  └─────────────┘  └──────────────┘  └────────────┘  │   │
│  │         │                  │                  │              │          │   │
│  │         └──────────────────┴──────────────────┴──────────────┘          │   │
│  │                               │                                         │   │
│  │                               ▼                                         │   │
│  │                  ┌──────────────────────────┐                          │   │
│  │                  │ Model Registry (MLflow)  │                          │   │
│  │                  │  - ResNet-18-v1.2.onnx   │                          │   │
│  │                  │  - ResNet-50-v2.0.onnx   │                          │   │
│  │                  └──────────────────────────┘                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                 │                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                   Inference Service (Real-time)                         │   │
│  │  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │   │
│  │  │ ONNX Runtime │->│ TensorRT    │->│ Prediction   │->│ Grad-CAM   │  │   │
│  │  │ (CPU/GPU)    │  │ (FP16/INT8) │  │ + Confidence │  │ Heatmap    │  │   │
│  │  └──────────────┘  └─────────────┘  └──────────────┘  └────────────┘  │   │
│  │         │                                     │              │          │   │
│  │         └─────────────────────────────────────┴──────────────┘          │   │
│  │                               │                                         │   │
│  │                               ▼                                         │   │
│  │                  ┌──────────────────────────┐                          │   │
│  │                  │ Redis Cache (Predictions)│                          │   │
│  │                  │  TTL: 1 hour             │                          │   │
│  │                  └──────────────────────────┘                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            API & Application Layer                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        FastAPI REST API                                 │   │
│  │  Endpoints: /predict, /grad-cam, /similarity-search, /retrain          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                 │                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      React Frontend (Web UI)                            │   │
│  │  - Wafer Map Viewer (zoom/pan)  - Grad-CAM Overlay                     │   │
│  │  - Prediction Dashboard          - Training Monitoring                 │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       Monitoring & Observability Layer                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐      │
│  │ Prometheus   │  │  Grafana    │  │ OpenSearch   │  │ Jaeger Tracing │      │
│  │ (Metrics)    │  │ (Dashboards)│  │ (Logs)       │  │ (Traces)       │      │
│  └──────────────┘  └─────────────┘  └──────────────┘  └────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Component Details

**STDF Parser & Wafer Map Generator**:
- **Input**: STDF files (binary format, 5-50MB per wafer depending on test count)
- **Parser**: pystdf library for STDF record extraction (PTR, MPR, FTR, PRR records)
- **Process**: 
  1. Extract device coordinates (x, y) and bin assignments from PRR (Part Result Record)
  2. Normalize coordinates to 300x300 grid (interpolate if die count < 90,000)
  3. Assign colors: PASS (green #00FF00), FAIL bins (red #FF0000, orange #FF8C00, yellow #FFD700), NOTEST (gray #808080)
  4. Render RGB image with PIL/OpenCV
- **Output**: 300x300 RGB PNG image (50-200KB compressed), metadata JSON (lot, wafer, die_count, bin_summary)
- **Performance**: <30 seconds per wafer (5,000 dies)

**ResNet Training Pipeline**:
- **DataLoader**: PyTorch DataLoader with 4-8 workers, batch_size=32-64 (depends on GPU memory)
- **Data Augmentation**: albumentations pipeline applied on-the-fly during training
- **Training Loop**: PyTorch Lightning Trainer with:
  - Distributed Data Parallel (DDP) for multi-GPU
  - Automatic Mixed Precision (AMP) for 2× speedup
  - Gradient accumulation (4 steps) for effective batch_size=256
  - EarlyStopping callback (patience=5, monitor='val_accuracy')
  - ModelCheckpoint callback (save_top_k=3, monitor='val_accuracy')
- **Logging**: MLflow autolog for hyperparameters, metrics, model artifacts

**ONNX Inference Service**:
- **Model Loading**: ONNX Runtime with TensorRT execution provider (GPU) or OpenVINO (CPU)
- **Preprocessing**: Resize to 224×224, normalize with ImageNet stats ([0.485,0.456,0.406], [0.229,0.224,0.225])
- **Batch Inference**: Dynamic batching (collect requests for 50ms, batch up to 64, process together)
- **Postprocessing**: Softmax for classification, sigmoid for regression, temperature scaling for confidence calibration
- **Caching**: Redis cache with key=hash(wafer_map_image), value=prediction+confidence, TTL=1 hour

**FastAPI REST API**:
- **Endpoints**:
  - `POST /api/v1/predict`: Upload STDF or wafer map image → yield prediction + defect class + confidence
  - `GET /api/v1/grad-cam/{wafer_id}`: Retrieve Grad-CAM heatmap for wafer
  - `POST /api/v1/similarity-search`: Find similar historical wafer maps (top-10, cosine similarity >0.85)
  - `POST /api/v1/retrain`: Trigger model retraining (admin only)
- **Authentication**: OAuth2 with JWT tokens, role-based access control (RBAC)
- **Rate Limiting**: 100 requests/minute per user, 1,000 requests/minute aggregate
- **Async Processing**: Background tasks for long-running operations (retraining, batch predictions)

**React Frontend**:
- **Wafer Map Viewer**: 
  - React-Konva canvas for zoom/pan interactions
  - Hover tooltip: die coordinates, bin, test results
  - Click: drill down to die-level parametric data
- **Prediction Dashboard**: Plotly charts for yield trends, defect type distribution over time
- **Training Monitoring**: Real-time TensorBoard embedding for loss curves, learning rate schedules
- **Grad-CAM Overlay**: Toggle heatmap on/off, adjust opacity slider (0-100%)

**Monitoring Components**:
- **Prometheus**: Scrape metrics from FastAPI (/metrics endpoint), ONNX Runtime, GPU (nvidia-smi exporter)
- **Grafana**: Dashboards for latency (p50/p95/p99), throughput (requests/sec), GPU utilization (%), model accuracy (daily)
- **OpenSearch**: Centralized logs with correlation IDs, searchable by wafer_id, user_id, error_type
- **Jaeger**: Distributed traces showing STDF parse → wafer map gen → inference → response (end-to-end latency breakdown)

### 9.3 Data Flow

**Training Data Flow**:
1. **STDF Collection**: Testers write STDF files to shared FTP server or S3 bucket (continuous, ~1,000 wafers/day)
2. **Batch Processing**: Nightly cron job triggers STDF parser (process previous day's data)
3. **Wafer Map Generation**: Convert STDF → 300×300 PNG images, store in MinIO with metadata in Parquet
4. **Labeling**: Yield engineers review wafer maps via UI, assign defect type labels (normal, edge, center, ring, etc.)
5. **Dataset Versioning**: DVC tracks wafer map dataset versions (train/val/test splits, 70/15/15)
6. **Model Training**: Weekly or monthly retraining job (triggered manually or by data drift detection)
   - Load dataset from MinIO
   - Apply progressive fine-tuning (Phase 1 → Phase 2 → Phase 3)
   - Log to MLflow (hyperparameters, metrics, model artifacts)
   - Save best checkpoint to model registry
7. **Model Validation**: Holdout test set evaluation, compare to production model baseline
8. **Model Deployment**: If new model accuracy > production + 2%, promote to staging, run A/B test for 1 week, then production

**Inference Data Flow (Real-time)**:
1. **Trigger**: User uploads STDF file via UI or API receives wafer_map_id from production MES
2. **Wafer Map Lookup**: Check if wafer map already exists in MinIO (by wafer_id hash)
   - If exists: retrieve from MinIO
   - If not: generate on-the-fly from STDF (30 seconds)
3. **Cache Check**: Query Redis cache with key=hash(wafer_map_image)
   - If hit: return cached prediction (latency <10ms)
   - If miss: proceed to inference
4. **Preprocessing**: Resize to 224×224, normalize, convert to tensor
5. **Inference**: ONNX Runtime + TensorRT prediction on GPU (latency <200ms)
6. **Postprocessing**: Apply temperature scaling, compute Grad-CAM (if requested)
7. **Cache Update**: Store prediction in Redis (TTL=1 hour)
8. **Response**: Return JSON with yield, defect_class, confidence, grad_cam_url
9. **Async Logging**: Background task logs prediction to PostgreSQL (wafer_id, timestamp, prediction, model_version)
10. **Monitoring**: Emit Prometheus metrics (latency, throughput), log to OpenSearch (structured JSON)

**Batch Inference Data Flow (Offline)**:
1. **Trigger**: User selects lot (100-300 wafers) or date range for batch prediction
2. **Job Submission**: API creates batch job in queue (Celery or Kubernetes Job)
3. **Parallel Processing**: Worker pods process wafers in parallel (10 workers × 100 wafers/hour = 1,000 wafers/hour)
4. **Results Aggregation**: Collect predictions, compute lot-level statistics (avg yield, defect type distribution)
5. **Report Generation**: Generate PDF report with wafer map gallery, Grad-CAM heatmaps, yield trends
6. **Notification**: Email/Slack notification when job completes (typically 15-30 minutes for 300 wafers)

---

## 10. Data Model

### 10.1 Entity Relationships

```
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│   Product    │──────<│     Lot      │──────<│    Wafer     │
│              │ 1:N   │              │ 1:N   │              │
│ product_id   │       │ lot_id       │       │ wafer_id     │
│ product_name │       │ product_id   │       │ lot_id       │
│ package_type │       │ start_date   │       │ wafer_num    │
│ die_count    │       │ end_date     │       │ die_count    │
└──────────────┘       │ target_yield │       │ stdf_path    │
                       └──────────────┘       │ wafer_map_url│
                                              └──────────────┘
                                                     │
                                                     │ 1:N
                                                     ▼
                                              ┌──────────────┐
                                              │  Prediction  │
                                              │              │
                                              │ prediction_id│
                                              │ wafer_id     │
                                              │ model_version│
                                              │ yield_pred   │
                                              │ defect_class │
                                              │ confidence   │
                                              │ timestamp    │
                                              └──────────────┘
                                                     │
                                                     │ 1:1
                                                     ▼
                                              ┌──────────────┐
                                              │  GradCAM     │
                                              │              │
                                              │ gradcam_id   │
                                              │ prediction_id│
                                              │ heatmap_url  │
                                              │ layer_name   │
                                              └──────────────┘

┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│ Model        │──────<│ Experiment   │──────<│ ModelMetric  │
│              │ 1:N   │              │ 1:N   │              │
│ model_id     │       │ experiment_id│       │ metric_id    │
│ model_name   │       │ model_id     │       │ experiment_id│
│ architecture │       │ run_id       │       │ metric_name  │
│ version      │       │ hyperparams  │       │ metric_value │
│ onnx_path    │       │ start_time   │       │ step         │
│ stage        │       │ end_time     │       └──────────────┘
│ created_at   │       │ status       │
└──────────────┘       └──────────────┘

┌──────────────┐       ┌──────────────┐
│  Dataset     │──────<│ WaferMapImg  │
│              │ 1:N   │              │
│ dataset_id   │       │ image_id     │
│ dataset_name │       │ wafer_id     │
│ version      │       │ dataset_id   │
│ split_type   │       │ image_url    │
│ created_at   │       │ defect_label │
│ dvc_hash     │       │ yield_actual │
└──────────────┘       └──────────────┘
```

### 10.2 Database Schema

**PostgreSQL Tables** (Relational Metadata):

```sql
-- Products Table
CREATE TABLE products (
    product_id VARCHAR(50) PRIMARY KEY,
    product_name VARCHAR(100) NOT NULL,
    package_type VARCHAR(20) NOT NULL,  -- BGA436, BGA292, etc.
    die_count INT NOT NULL,
    max_x INT NOT NULL,
    max_y INT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Lots Table
CREATE TABLE lots (
    lot_id VARCHAR(50) PRIMARY KEY,
    product_id VARCHAR(50) REFERENCES products(product_id),
    fab_site VARCHAR(20),
    start_date DATE,
    end_date DATE,
    target_yield DECIMAL(5,2),
    actual_yield DECIMAL(5,2),
    status VARCHAR(20)  -- IN_PROGRESS, COMPLETED, HELD
);

-- Wafers Table
CREATE TABLE wafers (
    wafer_id VARCHAR(50) PRIMARY KEY,
    lot_id VARCHAR(50) REFERENCES lots(lot_id),
    wafer_num INT NOT NULL,
    die_count INT,
    pass_count INT,
    fail_count INT,
    stdf_path VARCHAR(500),
    wafer_map_url VARCHAR(500),
    test_completion_pct DECIMAL(5,2),  -- 5%, 10%, 100%
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_lot_wafer (lot_id, wafer_num),
    INDEX idx_test_completion (test_completion_pct)
);

-- Predictions Table
CREATE TABLE predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    wafer_id VARCHAR(50) REFERENCES wafers(wafer_id),
    model_version VARCHAR(50) NOT NULL,  -- resnet18-v1.2, resnet50-v2.0
    yield_pred DECIMAL(5,2) NOT NULL,
    defect_class VARCHAR(50),  -- EdgeEffect, CenterCluster, etc.
    confidence DECIMAL(5,4),  -- 0.0-1.0
    inference_time_ms INT,  -- latency in milliseconds
    timestamp TIMESTAMP DEFAULT NOW(),
    user_id VARCHAR(50),
    INDEX idx_wafer_model (wafer_id, model_version),
    INDEX idx_timestamp (timestamp)
);

-- GradCAM Heatmaps Table
CREATE TABLE gradcam_heatmaps (
    gradcam_id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT REFERENCES predictions(prediction_id),
    heatmap_url VARCHAR(500),
    layer_name VARCHAR(50),  -- layer4, layer3, layer2
    created_at TIMESTAMP DEFAULT NOW()
);

-- Models Table (Model Registry)
CREATE TABLE models (
    model_id VARCHAR(50) PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,  -- ResNet-18 Transfer Learning
    architecture VARCHAR(50),  -- resnet18, resnet50
    version VARCHAR(20),  -- v1.0, v1.1, v2.0
    onnx_path VARCHAR(500),
    pytorch_path VARCHAR(500),
    stage VARCHAR(20),  -- STAGING, PRODUCTION, ARCHIVED
    accuracy DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(50)
);

-- Experiments Table (MLflow Integration)
CREATE TABLE experiments (
    experiment_id VARCHAR(50) PRIMARY KEY,
    model_id VARCHAR(50) REFERENCES models(model_id),
    run_id VARCHAR(100) UNIQUE,  -- MLflow run_id
    hyperparams JSONB,  -- {lr: 0.001, batch_size: 32, epochs: 10}
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    status VARCHAR(20),  -- RUNNING, COMPLETED, FAILED
    training_samples INT,
    validation_samples INT
);

-- Model Metrics Table
CREATE TABLE model_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    experiment_id VARCHAR(50) REFERENCES experiments(experiment_id),
    metric_name VARCHAR(50),  -- val_accuracy, val_loss, train_loss
    metric_value DECIMAL(10,6),
    step INT,  -- epoch number or global step
    timestamp TIMESTAMP DEFAULT NOW(),
    INDEX idx_experiment_metric (experiment_id, metric_name)
);

-- Datasets Table
CREATE TABLE datasets (
    dataset_id VARCHAR(50) PRIMARY KEY,
    dataset_name VARCHAR(100),
    version VARCHAR(20),
    split_type VARCHAR(10),  -- TRAIN, VAL, TEST
    sample_count INT,
    created_at TIMESTAMP DEFAULT NOW(),
    dvc_hash VARCHAR(100),  -- DVC commit hash for reproducibility
    description TEXT
);

-- WaferMap Images Table
CREATE TABLE wafermap_images (
    image_id BIGSERIAL PRIMARY KEY,
    wafer_id VARCHAR(50) REFERENCES wafers(wafer_id),
    dataset_id VARCHAR(50) REFERENCES datasets(dataset_id),
    image_url VARCHAR(500),
    defect_label VARCHAR(50),  -- manual label by yield engineer
    yield_actual DECIMAL(5,2),
    resolution VARCHAR(20),  -- 300x300, 1024x1024
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_dataset_label (dataset_id, defect_label)
);
```

**Parquet Tables** (Analytical Data in MinIO/S3):

```
wafer_test_results/
├── lot_id=L12345/
│   ├── wafer_id=W001.parquet
│   ├── wafer_id=W002.parquet
│   └── ...
└── lot_id=L12346/
    └── ...

Schema:
- die_x: INT (0-99)
- die_y: INT (0-99)
- bin: INT (1-255)
- test_results: STRUCT<IDDQ FLOAT, Vth FLOAT, Fmax FLOAT, ...>
- timestamp: TIMESTAMP

prediction_history/
├── date=2025-12-01/
│   ├── model_version=resnet18-v1.2/
│   │   └── predictions.parquet
│   └── model_version=resnet50-v2.0/
│       └── predictions.parquet
└── date=2025-12-02/
    └── ...

Schema:
- wafer_id: STRING
- yield_pred: FLOAT
- defect_class: STRING
- confidence: FLOAT
- inference_time_ms: INT
- timestamp: TIMESTAMP
```

### 10.3 Data Flow Diagrams

**Wafer Map Generation Flow**:
```
STDF File → Parser → Coordinates + Bins → Normalize (300x300) → Render RGB → MinIO
   │             │           │                   │                   │          │
   │             │           │                   │                   │          └→ wafer_map_url
   │             │           │                   │                   └→ PNG (50-200KB)
   │             │           │                   └→ (x,y) → (i,j) grid mapping
   │             │           └→ {die_1: (x,y,bin), die_2: (x,y,bin), ...}
   │             └→ PRR records extraction
   └→ 5-50MB binary

Metadata → PostgreSQL wafers table (wafer_id, die_count, pass_count, wafer_map_url)
```

**Prediction Request Flow**:
```
API Request → Redis Cache Check → [Cache Hit] → Return Cached Prediction (10ms)
   │                  │
   │                  └→ [Cache Miss] → Wafer Map Retrieval (MinIO)
   │                                           │
   └→ wafer_id                                 ▼
                                         Preprocessing (224x224, normalize)
                                               │
                                               ▼
                                         ONNX Inference (GPU, 200ms)
                                               │
                                               ▼
                                         Postprocessing (softmax, calibration)
                                               │
                                               ├→ Prediction (yield, defect, confidence)
                                               ├→ Cache Update (Redis, TTL=1hr)
                                               ├→ Log to PostgreSQL predictions table
                                               └→ Return to API
```

**Training Pipeline Flow**:
```
DVC Dataset → DataLoader (augmentation) → ResNet Model → Loss Calculation → Backprop
   │              │                           │               │                │
   │              │                           │               │                └→ Optimizer Step
   │              │                           │               └→ CrossEntropyLoss + MSE
   │              │                           └→ Phase 1/2/3 (progressive fine-tuning)
   │              └→ {train: 70%, val: 15%, test: 15%}
   └→ Versioned wafer maps (train_v1.0, train_v1.1, ...)

After each epoch:
├→ MLflow Log (loss, accuracy, learning_rate)
├→ Model Checkpoint (save if val_accuracy improved)
└→ Validation Evaluation

After training:
├→ Test Set Evaluation → Final Metrics
├→ Model Registration (MLflow Model Registry, stage=STAGING)
└→ ONNX Export → TensorRT Optimization → MinIO Storage
```

### 10.4 Input Data & Dataset Requirements

**STDF Files (Input)**:
- **Format**: Standard Test Data Format (IEEE 1671), binary
- **Size**: 5-50MB per wafer (depends on test count: 1,000-10,000 tests per die)
- **Frequency**: Continuous (1,000 wafers/day from production)
- **Retention**: 90 days on FTP server, 2 years in S3 archive
- **Records Used**: PRR (Part Result Record) for (x, y, bin), PTR (Parametric Test Record) for test values

**Wafer Map Images (Processed)**:
- **Format**: PNG (RGB), 300×300 pixels
- **Size**: 50-200KB compressed per image
- **Total Dataset**: 
  - Training: 50,000 wafers (10GB)
  - Validation: 10,000 wafers (2GB)
  - Test: 10,000 wafers (2GB)
- **Labeling**: 
  - Automated: bin-based coloring (PASS/FAIL)
  - Manual: defect type labels for 10,000 wafers (8 classes: edge, center, ring, quadrant, scratch, random, mixed, normal)
  - Labeling tool: Custom React UI with keyboard shortcuts for fast annotation

**Dataset Splits**:
- **Temporal Split**: Train on 2024 Q1-Q3, validate on 2024 Q4, test on 2025 Q1 (avoid data leakage)
- **Product Split**: Ensure all products represented in train/val/test (stratified sampling)
- **Defect Type Split**: Balance defect types in validation set (avoid class imbalance)

**Data Augmentation** (Training Only):
- Rotations: 0°, 90°, 180°, 270° (4× data multiplication)
- Flips: Horizontal, vertical (2× data multiplication)
- Combined: 4 rotations × 2 flips = 8× effective training data
- Color jitter: ±20% brightness, ±20% contrast (applied with p=0.5)
- **Total Effective Training Data**: 50,000 × 8 = 400,000 augmented samples

**Data Versioning (DVC)**:
- **Version 1.0**: Initial dataset (50K train, 10K val, 10K test) - Dec 2024
- **Version 1.1**: Added 10K new samples, rebalanced defect types - Mar 2025
- **Version 2.0**: Multi-product dataset (TC41x + TC42x + TC43x) - Jun 2025
- **DVC Commands**:
  ```
  dvc add data/wafer_maps/train_v1.0
  dvc add data/wafer_maps/val_v1.0
  dvc add data/wafer_maps/test_v1.0
  git add data/wafer_maps/train_v1.0.dvc
  git commit -m "Add wafer map dataset v1.0"
  dvc push  # Upload to S3/MinIO remote storage
  ```

---

## 11. API Specifications

### 11.1 REST Endpoints

**Prediction API**:

```http
POST /api/v1/predict
Content-Type: multipart/form-data
Authorization: Bearer <JWT_TOKEN>

Parameters:
- stdf_file: File (STDF binary) OR wafer_map_image: File (PNG/JPEG)
- product_id: string (optional, for model selection)
- test_completion_pct: float (optional, 5.0, 10.0, 20.0, 100.0)
- include_gradcam: boolean (default: false)
- gradcam_layer: string (default: "layer4", options: layer2, layer3, layer4)

Response (200 OK):
{
  "wafer_id": "W12345-001",
  "prediction": {
    "yield": 87.3,
    "defect_class": "EdgeEffect",
    "defect_probabilities": {
      "EdgeEffect": 0.72,
      "CenterCluster": 0.15,
      "Normal": 0.08,
      "RingPattern": 0.03,
      "QuadrantFailure": 0.01,
      "Scratch": 0.01,
      "RandomFailure": 0.00,
      "MixedMode": 0.00
    },
    "confidence": 0.92,
    "uncertainty": 0.08
  },
  "model_version": "resnet18-v1.2",
  "inference_time_ms": 187,
  "grad_cam_url": "https://minio.example.com/gradcam/W12345-001_layer4.png",  # if requested
  "timestamp": "2025-12-04T10:30:45Z"
}

Error Responses:
- 400 Bad Request: Invalid file format, missing parameters
- 401 Unauthorized: Invalid or expired JWT token
- 413 Payload Too Large: File size > 100MB
- 500 Internal Server Error: Model inference failed
```

**Batch Prediction API**:

```http
POST /api/v1/predict/batch
Content-Type: application/json
Authorization: Bearer <JWT_TOKEN>

Request Body:
{
  "wafer_ids": ["W12345-001", "W12345-002", "W12345-003"],
  "lot_id": "L12345",  # optional, process all wafers in lot
  "include_gradcam": false,
  "model_version": "resnet18-v1.2"  # optional, default to production model
}

Response (202 Accepted):
{
  "job_id": "batch_20251204_103045",
  "status": "QUEUED",
  "total_wafers": 3,
  "estimated_time_seconds": 180,
  "status_url": "/api/v1/jobs/batch_20251204_103045"
}

Status Check:
GET /api/v1/jobs/{job_id}
Response:
{
  "job_id": "batch_20251204_103045",
  "status": "COMPLETED",  # QUEUED, RUNNING, COMPLETED, FAILED
  "progress": {
    "completed": 3,
    "total": 3,
    "percentage": 100.0
  },
  "results_url": "/api/v1/results/batch_20251204_103045",
  "created_at": "2025-12-04T10:30:45Z",
  "completed_at": "2025-12-04T10:33:52Z"
}
```

**Similarity Search API**:

```http
POST /api/v1/similarity-search
Content-Type: application/json
Authorization: Bearer <JWT_TOKEN>

Request Body:
{
  "wafer_id": "W12345-001",  # OR wafer_map_image (base64 encoded)
  "top_k": 10,
  "min_similarity": 0.85,  # cosine similarity threshold
  "filters": {
    "product_id": "TC42x",  # optional
    "defect_class": "EdgeEffect",  # optional
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-12-31"
    }
  }
}

Response (200 OK):
{
  "query_wafer_id": "W12345-001",
  "results": [
    {
      "wafer_id": "W11234-005",
      "similarity": 0.93,
      "defect_class": "EdgeEffect",
      "yield_actual": 85.2,
      "lot_id": "L11234",
      "test_date": "2024-11-15",
      "wafer_map_url": "https://minio.example.com/wafer_maps/W11234-005.png"
    },
    {
      "wafer_id": "W10987-012",
      "similarity": 0.89,
      "defect_class": "EdgeEffect",
      "yield_actual": 88.1,
      "lot_id": "L10987",
      "test_date": "2024-10-20",
      "wafer_map_url": "https://minio.example.com/wafer_maps/W10987-012.png"
    },
    # ... 8 more results
  ],
  "total_results": 10,
  "search_time_ms": 45
}
```

**Model Management API**:

```http
GET /api/v1/models
Authorization: Bearer <JWT_TOKEN>

Response (200 OK):
{
  "models": [
    {
      "model_id": "resnet18-v1.2",
      "model_name": "ResNet-18 Transfer Learning (TC42x)",
      "architecture": "resnet18",
      "version": "v1.2",
      "stage": "PRODUCTION",
      "accuracy": 0.9245,
      "created_at": "2025-11-15T14:20:00Z",
      "created_by": "ml_engineer@example.com"
    },
    {
      "model_id": "resnet50-v2.0",
      "model_name": "ResNet-50 Transfer Learning (Multi-Product)",
      "architecture": "resnet50",
      "version": "v2.0",
      "stage": "STAGING",
      "accuracy": 0.9387,
      "created_at": "2025-12-01T09:15:00Z",
      "created_by": "ml_engineer@example.com"
    }
  ]
}

POST /api/v1/models/{model_id}/promote
Authorization: Bearer <JWT_TOKEN> (admin role required)

Request Body:
{
  "target_stage": "PRODUCTION",  # STAGING → PRODUCTION
  "rollback_on_degradation": true,  # auto-rollback if accuracy drops >3%
  "ab_test_duration_hours": 168  # 1 week A/B test before full rollout
}

Response (200 OK):
{
  "model_id": "resnet50-v2.0",
  "stage": "PRODUCTION",
  "previous_model": "resnet18-v1.2",
  "ab_test_start": "2025-12-04T10:30:00Z",
  "ab_test_end": "2025-12-11T10:30:00Z",
  "status": "AB_TESTING"
}
```

**Retraining API**:

```http
POST /api/v1/retrain
Content-Type: application/json
Authorization: Bearer <JWT_TOKEN> (admin role required)

Request Body:
{
  "dataset_version": "v1.1",  # DVC dataset version
  "architecture": "resnet18",  # resnet18 or resnet50
  "hyperparameters": {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 15,
    "freeze_backbone": true,  # Phase 1: freeze
    "unfreeze_last_block": true,  # Phase 2: unfreeze layer4
    "full_finetuning": true  # Phase 3: full network
  },
  "product_id": "TC42x",  # optional, train product-specific model
  "priority": "HIGH"  # HIGH, NORMAL, LOW (queue priority)
}

Response (202 Accepted):
{
  "training_job_id": "train_20251204_103045",
  "status": "QUEUED",
  "estimated_duration_hours": 18,
  "gpu_allocated": "A100",
  "status_url": "/api/v1/training-jobs/train_20251204_103045"
}

GET /api/v1/training-jobs/{job_id}
Response:
{
  "training_job_id": "train_20251204_103045",
  "status": "RUNNING",  # QUEUED, RUNNING, COMPLETED, FAILED
  "progress": {
    "current_epoch": 5,
    "total_epochs": 15,
    "phase": "Phase 2: Unfreeze Last Block",
    "train_loss": 0.342,
    "val_accuracy": 0.8876,
    "elapsed_hours": 6.2,
    "eta_hours": 11.8
  },
  "mlflow_url": "http://mlflow.example.com/experiments/12/runs/abc123",
  "tensorboard_url": "http://tensorboard.example.com/?run=train_20251204_103045"
}
```

### 11.2 Request/Response Examples

**Example 1: Upload STDF and Get Prediction**

```bash
curl -X POST "https://api.example.com/api/v1/predict" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -F "stdf_file=@/path/to/wafer_W12345-001.stdf" \
  -F "product_id=TC42x" \
  -F "test_completion_pct=10.0" \
  -F "include_gradcam=true" \
  -F "gradcam_layer=layer4"
```

Response:
```json
{
  "wafer_id": "W12345-001",
  "prediction": {
    "yield": 87.3,
    "defect_class": "EdgeEffect",
    "defect_probabilities": {
      "EdgeEffect": 0.72,
      "CenterCluster": 0.15,
      "Normal": 0.08,
      "RingPattern": 0.03,
      "QuadrantFailure": 0.01,
      "Scratch": 0.01,
      "RandomFailure": 0.00,
      "MixedMode": 0.00
    },
    "confidence": 0.92,
    "uncertainty": 0.08
  },
  "model_version": "resnet18-v1.2",
  "inference_time_ms": 187,
  "grad_cam_url": "https://minio.example.com/gradcam/W12345-001_layer4.png",
  "timestamp": "2025-12-04T10:30:45Z"
}
```

**Example 2: Batch Prediction for Entire Lot**

```python
import requests

api_url = "https://api.example.com"
jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# Submit batch job
response = requests.post(
    f"{api_url}/api/v1/predict/batch",
    headers={"Authorization": f"Bearer {jwt_token}"},
    json={
        "lot_id": "L12345",
        "include_gradcam": False,
        "model_version": "resnet18-v1.2"
    }
)
job_data = response.json()
job_id = job_data["job_id"]
print(f"Job submitted: {job_id}, ETA: {job_data['estimated_time_seconds']}s")

# Poll for status
import time
while True:
    status_response = requests.get(
        f"{api_url}/api/v1/jobs/{job_id}",
        headers={"Authorization": f"Bearer {jwt_token}"}
    )
    status_data = status_response.json()
    print(f"Progress: {status_data['progress']['percentage']:.1f}%")
    
    if status_data["status"] == "COMPLETED":
        results_url = status_data["results_url"]
        print(f"Job completed! Results: {results_url}")
        break
    elif status_data["status"] == "FAILED":
        print(f"Job failed!")
        break
    
    time.sleep(10)  # Poll every 10 seconds
```

**Example 3: Find Similar Wafer Maps**

```javascript
// JavaScript (React frontend)
const searchSimilar = async (waferMapFile) => {
  const base64Image = await fileToBase64(waferMapFile);
  
  const response = await fetch('/api/v1/similarity-search', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${jwtToken}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      wafer_map_image: base64Image,
      top_k: 10,
      min_similarity: 0.85,
      filters: {
        product_id: 'TC42x',
        date_range: {
          start: '2024-01-01',
          end: '2024-12-31'
        }
      }
    })
  });
  
  const data = await response.json();
  console.log(`Found ${data.total_results} similar wafer maps:`);
  data.results.forEach((result, index) => {
    console.log(`${index + 1}. Wafer ${result.wafer_id}, Similarity: ${result.similarity}, Yield: ${result.yield_actual}%`);
  });
  
  return data.results;
};
```

### 11.3 Authentication

**OAuth2 + JWT Flow**:

1. **User Login**: Redirect to OAuth2 provider (Azure AD, Okta, Auth0)
2. **Authorization**: User authenticates, provider returns authorization code
3. **Token Exchange**: Backend exchanges code for access_token + refresh_token
4. **JWT Generation**: Backend generates JWT with claims (user_id, roles, exp)
5. **API Requests**: Client includes JWT in `Authorization: Bearer <token>` header
6. **Token Validation**: API validates JWT signature, expiration, roles before processing request

**JWT Claims**:
```json
{
  "sub": "user123",
  "email": "engineer@example.com",
  "roles": ["yield_engineer", "model_user"],
  "product_access": ["TC41x", "TC42x", "TC43x"],
  "iat": 1701691200,
  "exp": 1701777600
}
```

**Role-Based Access Control (RBAC)**:
- **model_user**: Can call /predict, /similarity-search, view wafer maps
- **yield_engineer**: model_user + can label wafer maps, trigger retraining
- **ml_engineer**: yield_engineer + can promote models, modify hyperparameters
- **admin**: Full access including user management, system configuration

**Token Refresh**:
```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}

Response (200 OK):
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

---

## 12. UI/UX Requirements

### 12.1 User Interface

**Wafer Map Viewer**:
- **Canvas Rendering**: React-Konva or HTML5 Canvas for high-performance wafer map visualization
- **Zoom & Pan**: Mouse wheel zoom (2×, 5×, 10×), click-drag pan, double-click to reset
- **Die-Level Details**: Hover tooltip showing (x, y), bin, PASS/FAIL status, key parametric values (IDDQ, Vth, Fmax)
- **Color Schemes**: 
  - Default: PASS (green), FAIL (red gradients by bin 2-9), NOTEST (gray)
  - Colorblind-friendly: PASS (blue), FAIL (orange/yellow), NOTEST (gray)
  - Custom: User-defined color mapping for specific bin analysis
- **Overlay Layers**: Toggle Grad-CAM heatmap, prediction confidence zones, defect pattern annotations
- **Export**: Download wafer map as PNG (high-res 1024×1024), PDF report, or raw data (CSV)

**Prediction Dashboard**:
- **Real-Time Updates**: WebSocket connection for live predictions as wafers complete testing
- **Prediction Card**: Display wafer ID, predicted yield (large font), defect class badge, confidence meter (0-100%)
- **Trend Charts**: 
  - Yield prediction over time (last 7/30/90 days) with Plotly line charts
  - Defect type distribution pie chart (8 categories)
  - Prediction accuracy vs. actual yield scatter plot
  - Confidence calibration reliability diagram
- **Filtering**: Filter by product, date range, defect type, yield range (slider: 0-100%), confidence threshold
- **Sorting**: Sort wafers by predicted yield (low to high), confidence (low to high), timestamp (newest first)

**Model Training Monitoring**:
- **Training Progress**: Real-time progress bar showing current epoch, phase (1/2/3), ETA
- **Loss Curves**: Embedded TensorBoard showing train/val loss, accuracy over epochs
- **Hyperparameters Panel**: Display learning rate schedule, batch size, optimizer settings
- **Checkpoint History**: List of saved checkpoints with val_accuracy, epoch number, timestamp
- **Compare Experiments**: Side-by-side comparison of multiple training runs (accuracy, loss, training time)

**Admin Panel**:
- **Model Management**: List all models (STAGING, PRODUCTION, ARCHIVED), promote/demote, view metadata
- **Dataset Management**: DVC dataset versions, upload new labeled data, view split statistics
- **User Management**: RBAC settings, add/remove users, assign roles (model_user, yield_engineer, ml_engineer, admin)
- **System Metrics**: GPU utilization, API request rates, error logs, storage usage

**Wafer Map Gallery**:
- **Thumbnail Grid**: 4×4 or 6×6 grid of wafer map thumbnails for quick browsing
- **Batch Selection**: Multi-select wafers for batch prediction, export, or labeling
- **Search & Filter**: Full-text search by lot ID, wafer ID, date range, defect type
- **Sort Options**: By yield (predicted or actual), confidence, timestamp, similarity to selected wafer

### 12.2 User Experience

**Onboarding Flow** (New User First Login):
1. **Welcome Screen**: Brief introduction to Transfer Learning Yield Predictor, value proposition
2. **Guided Tour**: Interactive tutorial (Shepherd.js) highlighting key features (wafer map upload, prediction view, Grad-CAM)
3. **Sample Dataset**: Pre-loaded sample wafers for experimentation (3 wafers: edge defect, center cluster, normal)
4. **First Prediction**: Step-by-step wizard to upload STDF → view prediction → interpret Grad-CAM
5. **Completion**: Badge unlocked, link to full documentation and video tutorials
6. **Target**: User completes onboarding in <30 minutes, ready for basic usage

**Prediction Workflow** (Common Use Case):
1. **Upload**: Drag-drop STDF file or select from recent wafers dropdown
2. **Auto-Processing**: Show progress spinner "Parsing STDF... Generating wafer map... Running inference..."
3. **Results Display**: Prediction card appears with yield, defect class, confidence
4. **Interpretation**: Click "Explain Prediction" → Grad-CAM heatmap overlay appears
5. **Action**: Options: "Mark for FA", "Add to Report", "Find Similar Wafers", "Re-run with Different Model"
6. **Feedback Loop**: Optional "Was this prediction helpful? Yes/No" for model improvement tracking
7. **Target**: End-to-end workflow <2 minutes for experienced users

**Batch Prediction Workflow**:
1. **Selection**: Select lot from dropdown or upload CSV with wafer IDs
2. **Configuration**: Choose model version, enable/disable Grad-CAM, set priority
3. **Submit**: Click "Start Batch Prediction" → Job queued
4. **Monitoring**: Progress bar shows completed/total wafers, live updates via WebSocket
5. **Notification**: Toast notification when job completes, email/Slack optional
6. **Results**: Auto-navigate to results page with sortable table, downloadable report (PDF/Excel)
7. **Target**: Minimal manual intervention, set-and-forget for large batches

**Model Retraining Workflow** (ML Engineer):
1. **Dataset Selection**: Choose DVC dataset version from dropdown, view stats (train/val/test counts)
2. **Hyperparameters**: Pre-filled form with recommended defaults, advanced options collapsible
3. **Resource Allocation**: Select GPU type (A10/A100), number of GPUs, priority (HIGH/NORMAL/LOW)
4. **Preview**: Show estimated training time, cost estimate, expected accuracy improvement
5. **Submit**: Click "Start Training" → Job queued, redirected to training monitoring page
6. **Monitoring**: Embedded TensorBoard, real-time metrics, ability to early-stop if diverging
7. **Completion**: Notification when training completes, auto-navigate to model comparison page
8. **Target**: ML engineer can configure and launch training in <10 minutes

**Error Handling & User Guidance**:
- **Graceful Degradation**: If model inference fails, show fallback message "Prediction temporarily unavailable, using rule-based estimate"
- **Actionable Errors**: Instead of "500 Internal Server Error", show "STDF file corrupted at record 1234. Please re-upload or contact support."
- **Progress Feedback**: Long-running operations (>5 seconds) show progress indicators, never block UI without feedback
- **Undo/Redo**: Support undo for destructive actions (delete wafer, archive model)
- **Confirmation Dialogs**: Confirm before critical actions (promote model to production, delete dataset)

**Accessibility & Inclusivity**:
- **Keyboard Navigation**: All features accessible via keyboard (Tab, Enter, Arrow keys, hotkeys)
- **Screen Reader Support**: ARIA labels for all interactive elements, semantic HTML
- **Color Contrast**: WCAG 2.1 AA compliance (4.5:1 for text, 3:1 for UI components)
- **Responsive Design**: Works on desktop (1920×1080), laptop (1366×768), tablet (768×1024), mobile view-only (no editing)
- **Font Sizing**: User-adjustable font size (100%, 125%, 150%), no horizontal scrolling
- **Focus Indicators**: Clear focus outline for keyboard navigation

### 12.3 Accessibility

**WCAG 2.1 Level AA Compliance**:
- **Perceivable**:
  - Text alternatives for wafer map images (alt text: "Wafer W12345-001, EdgeEffect defect, 87.3% yield")
  - Captions for video tutorials, transcripts for audio content
  - Colorblind-friendly palettes (avoid red/green only, add patterns/labels)
  - Sufficient color contrast (4.5:1 for normal text, 3:1 for large text/UI)
  
- **Operable**:
  - All functionality keyboard accessible (no mouse-only interactions)
  - No keyboard traps (can exit all modal dialogs with Esc)
  - Adjustable time limits (extend session timeout via button before expiration)
  - No content flashing >3 times per second (avoid seizure triggers)
  
- **Understandable**:
  - Consistent navigation (same header/sidebar across all pages)
  - Input validation with clear error messages ("Wafer ID must be 10 characters, format: LXXXX-YYY")
  - Help tooltips (?) for technical terms (e.g., "Grad-CAM: Gradient-weighted Class Activation Mapping")
  - Language attribute set (<html lang="en">)
  
- **Robust**:
  - Valid HTML5 markup (passes W3C validator)
  - Compatible with assistive technologies (JAWS, NVDA screen readers)
  - Graceful degradation for older browsers (fallback to static images if Canvas unsupported)

**Internationalization (i18n)**:
- **Primary Language**: English (en-US)
- **Additional Languages** (if global deployment): German (de-DE), Simplified Chinese (zh-CN), Japanese (ja-JP)
- **Date/Time Formats**: Locale-aware (US: MM/DD/YYYY, EU: DD/MM/YYYY, ISO 8601 for APIs)
- **Number Formats**: Locale-aware (US: 1,234.56, EU: 1.234,56)
- **Translation Infrastructure**: i18next or react-intl for frontend, gettext for backend

**Keyboard Shortcuts** (Power User Features):
- `Ctrl+U`: Upload STDF file
- `Ctrl+P`: Predict (run inference on loaded wafer)
- `Ctrl+G`: Toggle Grad-CAM overlay
- `Ctrl+F`: Find similar wafers
- `Ctrl+E`: Export current view
- `Ctrl+Z`: Undo last action
- `Ctrl+Shift+T`: Open training monitoring page
- `Esc`: Close modal dialogs
- `Tab / Shift+Tab`: Navigate between form fields
- `Arrow Keys`: Pan wafer map viewer

**Mobile Experience** (View-Only):
- **Responsive Breakpoints**: Desktop (>1200px), Tablet (768-1200px), Mobile (<768px)
- **Mobile-Specific UI**:
  - Simplified wafer map viewer (tap to zoom, pinch gestures)
  - Vertical scrolling for prediction dashboard (no horizontal scroll)
  - Bottom navigation bar (Home, Predictions, Wafers, Profile)
  - Swipe gestures (left/right to navigate wafers)
- **Performance**: Lazy loading for wafer map images, virtualized lists (react-window) for long lists
- **Offline Mode**: Service worker caches recently viewed wafers for offline viewing (no predictions offline)

---

## 13. Security Requirements

### 13.1 Authentication

**OAuth2 / OpenID Connect (OIDC)**:
- **Identity Providers**: Azure Active Directory (primary), Okta, Auth0 (alternatives)
- **Flow**: Authorization Code Flow with PKCE (Proof Key for Code Exchange) for enhanced security
- **SSO**: Single Sign-On with corporate credentials, no separate password management
- **MFA**: Enforce Multi-Factor Authentication (TOTP, SMS, biometric) for admin and ml_engineer roles
- **Session Management**: 
  - Access token expiration: 1 hour
  - Refresh token expiration: 7 days
  - Idle timeout: 2 hours (auto-logout after inactivity)
  - Concurrent sessions: Max 3 active sessions per user

**JWT Token Security**:
- **Signing Algorithm**: RS256 (RSA with SHA-256, asymmetric keys preferred over HS256)
- **Key Rotation**: Rotate signing keys every 90 days, maintain 2 previous keys for verification
- **Claims Validation**: Verify issuer (iss), audience (aud), expiration (exp), not-before (nbf), issued-at (iat)
- **Token Storage**: 
  - Frontend: HttpOnly, Secure, SameSite=Strict cookies (prevent XSS theft)
  - Mobile: Secure storage (iOS Keychain, Android Keystore)
  - Backend: Redis with TTL matching token expiration

**Password Policy** (if local accounts used):
- **Complexity**: Min 12 characters, uppercase, lowercase, digit, special character
- **History**: Cannot reuse last 5 passwords
- **Expiration**: 90-day expiration for privileged accounts (admin, ml_engineer)
- **Lockout**: 5 failed attempts → 15-minute lockout, CAPTCHA after 3 failed attempts
- **Hashing**: Argon2id (OWASP recommended), bcrypt as fallback, salted per-user

### 13.2 Authorization

**Role-Based Access Control (RBAC)**:
- **Roles**:
  - `model_user`: Read-only access to predictions, wafer maps, similarity search (Yield Engineers, Test Engineers)
  - `data_labeler`: model_user + label wafer maps with defect types (Yield Engineers)
  - `model_trainer`: data_labeler + trigger model retraining, view experiment results (ML Engineers, Product Engineers)
  - `model_admin`: model_trainer + promote/demote models, modify production configurations (Senior ML Engineers)
  - `system_admin`: Full access including user management, infrastructure settings (DevOps, System Admins)

- **Permissions Matrix**:
  | Action | model_user | data_labeler | model_trainer | model_admin | system_admin |
  |--------|-----------|--------------|---------------|-------------|--------------|
  | View predictions | ✓ | ✓ | ✓ | ✓ | ✓ |
  | Upload STDF | ✓ | ✓ | ✓ | ✓ | ✓ |
  | View Grad-CAM | ✓ | ✓ | ✓ | ✓ | ✓ |
  | Label wafer maps | ✗ | ✓ | ✓ | ✓ | ✓ |
  | Trigger retraining | ✗ | ✗ | ✓ | ✓ | ✓ |
  | View experiments | ✗ | ✗ | ✓ | ✓ | ✓ |
  | Promote models | ✗ | ✗ | ✗ | ✓ | ✓ |
  | Manage users | ✗ | ✗ | ✗ | ✗ | ✓ |
  | System config | ✗ | ✗ | ✗ | ✗ | ✓ |

**Attribute-Based Access Control (ABAC)** (Advanced):
- **Product-Level Access**: Users can only access wafers/predictions for products they're authorized for
  - Example: TC41x engineer cannot view TC42x data without explicit grant
- **Confidentiality Levels**: Some wafers marked "Confidential" (NPI, customer returns) require elevated access
- **Time-Based Access**: Temporary access grants (e.g., contractor access for 30 days)
- **Location-Based**: Restrict admin actions to corporate network (VPN required for remote access)

**API Rate Limiting** (Prevent Abuse):
- **User-Level Limits**:
  - model_user: 100 requests/minute, 5,000 requests/day
  - model_trainer: 500 requests/minute, 20,000 requests/day
  - model_admin: 1,000 requests/minute, unlimited daily
- **Endpoint-Specific Limits**:
  - `/predict`: 50 requests/minute (compute-intensive)
  - `/similarity-search`: 20 requests/minute (vector DB queries expensive)
  - `/retrain`: 5 requests/day (prevent accidental duplicate jobs)
- **Burst Allowance**: Allow short bursts (2× rate for 10 seconds) for legitimate use
- **429 Response**: "Rate limit exceeded. Retry after 60 seconds."

### 13.3 Data Protection

**Data Encryption**:
- **In Transit**: 
  - TLS 1.3 for all API endpoints (no TLS 1.0/1.1, deprecated)
  - Certificate pinning for mobile apps (prevent MITM attacks)
  - HSTS (HTTP Strict Transport Security) with max-age=31536000 (1 year)
- **At Rest**:
  - AES-256 encryption for wafer map images in MinIO/S3 (server-side encryption)
  - Database encryption: PostgreSQL TDE (Transparent Data Encryption) or disk-level encryption (LUKS, BitLocker)
  - Model checkpoints: AES-256 encryption for ONNX files, keys stored in HashiCorp Vault
  - Backup encryption: All backups encrypted before storage, separate key per backup

**Secrets Management**:
- **HashiCorp Vault**: 
  - Store database passwords, API keys, model encryption keys, OAuth client secrets
  - Dynamic secrets: Generate temporary database credentials (TTL=1 hour) for batch jobs
  - Audit logging: Track all secret access (who, when, which secret)
- **Kubernetes Secrets**: 
  - Encrypt etcd at rest (--encryption-provider-config)
  - Use external secrets operator to sync from Vault
  - No secrets in container environment variables (use mounted volumes)
- **Git Security**:
  - No secrets in source code (use .env files in .gitignore)
  - Pre-commit hooks to scan for secrets (git-secrets, TruffleHog)
  - Rotate secrets immediately if accidentally committed

**Data Minimization & Retention**:
- **Collection**: Only collect necessary data (wafer maps, test results, predictions), no PII
- **Retention**:
  - Wafer maps: 2 years in hot storage, 5 years in cold archive (S3 Glacier), then delete
  - Predictions: 1 year in PostgreSQL, 3 years in data lake (Parquet), then delete
  - Logs: 90 days in OpenSearch, 1 year in S3, then delete
  - Model checkpoints: Keep last 10 versions, archive production models indefinitely
- **Right to Deletion**: Support data deletion requests (GDPR compliance, though semiconductor data unlikely to contain PII)

**Data Anonymization** (for external sharing):
- **Wafer IDs**: Hash wafer IDs before sharing with third parties (SHA-256 with secret salt)
- **Product Names**: Replace with generic labels (Product A, Product B) in publications/papers
- **Coordinates**: Normalize die coordinates to relative positions (avoid leaking die size/layout)
- **Metadata Stripping**: Remove fab site, lot IDs, timestamps before sharing datasets externally

### 13.4 Compliance

**Industry Standards**:
- **ISO/IEC 27001**: Information Security Management System (ISMS) certification
- **SOC 2 Type II**: Service Organization Control (for SaaS deployments)
- **NIST Cybersecurity Framework**: Align security practices with NIST CSF categories (Identify, Protect, Detect, Respond, Recover)

**Regulatory Compliance**:
- **GDPR** (if EU users): 
  - Data processing agreements with cloud providers
  - Right to access, rectify, delete data (automated workflows)
  - Data breach notification within 72 hours
  - Privacy by design (minimize data collection)
- **CCPA** (California Consumer Privacy Act): Similar to GDPR for California users
- **Export Controls** (ITAR, EAR): Ensure AI/ML models don't violate export regulations if deployed internationally

**Audit Logging**:
- **Security Events**: Log all authentication attempts (success/failure), authorization failures, privilege escalations
- **Data Access**: Log all wafer map accesses, predictions, model downloads (who, when, what)
- **Model Changes**: Log model promotions, demotions, deletions (who, when, from/to stages)
- **System Changes**: Log configuration changes, user additions/deletions, role assignments
- **Log Storage**: Immutable logs (append-only, cannot be deleted by users), encrypted at rest
- **Log Retention**: 1 year in searchable format (OpenSearch), 7 years in cold archive (compliance)

**Vulnerability Management**:
- **Dependency Scanning**: Daily scans with Snyk, Dependabot for CVEs in Python/Node.js packages
- **Container Scanning**: Trivy scans for Docker images before deployment
- **SAST**: Static Application Security Testing (SonarQube, Semgrep) in CI/CD pipeline
- **DAST**: Dynamic Application Security Testing (OWASP ZAP) on staging environment weekly
- **Penetration Testing**: Annual pen test by third-party security firm
- **Patch SLA**: Critical vulnerabilities (CVSS >9) patched within 7 days, high (CVSS 7-9) within 30 days

**Security Training**:
- **Developers**: Annual secure coding training (OWASP Top 10, secure API design)
- **Users**: Phishing awareness training quarterly
- **Admins**: Incident response training, privilege escalation prevention
- **Contractors**: Security NDA, background checks, access revoked within 24 hours of contract end

---

## 14. Performance Requirements

### 14.1 Response Times

**API Latency SLOs** (Service Level Objectives):
- **Single Prediction** (`POST /api/v1/predict`):
  - p50 (median): <150ms
  - p95: <200ms
  - p99: <300ms
  - Target: 95% of requests meet p95 SLO (measured monthly)
  
- **Batch Prediction** (`POST /api/v1/predict/batch`):
  - Job submission response: <500ms (returns job_id immediately)
  - Processing throughput: 1,000 wafers/hour sustained
  - Individual wafer inference: <200ms p95 (same as single prediction)
  
- **Similarity Search** (`POST /api/v1/similarity-search`):
  - p50: <100ms (vector DB query optimized)
  - p95: <200ms
  - p99: <500ms
  - Supports 10M+ wafer embeddings in index
  
- **Wafer Map Generation** (STDF → PNG):
  - Small wafers (<2,000 dies): <10 seconds
  - Medium wafers (2,000-5,000 dies): <20 seconds
  - Large wafers (>5,000 dies): <30 seconds
  - Parallelized: 10 wafers processed simultaneously
  
- **Grad-CAM Heatmap**:
  - Generation: <500ms additional latency (computed on-demand)
  - Caching: Subsequent requests <50ms (served from Redis)

**Page Load Times** (Web UI):
- **Initial Page Load**: <2 seconds (first contentful paint)
- **Time to Interactive**: <3 seconds
- **Wafer Map Rendering**: <1 second for 300×300 image
- **Dashboard Refresh**: <500ms for chart updates
- **Search Results**: <300ms for autocomplete suggestions

**Model Training Performance**:
- **Phase 1** (Freeze Backbone): 1-2 epochs, 2-4 hours on A10 GPU
- **Phase 2** (Unfreeze Last Block): 2-3 epochs, 4-8 hours on A10 GPU
- **Phase 3** (Full Fine-Tuning): 5-10 epochs, 10-20 hours on A10 GPU
- **Total Training Time**: 16-32 hours (single A10 GPU), 8-16 hours (2× A10 GPUs)
- **Multi-GPU Scaling**: 1.8× speedup with 2 GPUs, 3.2× with 4 GPUs (communication overhead)

### 14.2 Throughput

**Inference Throughput**:
- **GPU Inference** (NVIDIA A10):
  - Batch size 1: 200 wafers/second (5ms per wafer)
  - Batch size 32: 1,600 wafers/second (0.625ms per wafer amortized, optimal throughput)
  - Batch size 64: 2,000 wafers/second (diminishing returns, memory constrained)
  - Optimal batch size: 32-48 (balance latency vs. throughput)
  
- **CPU Inference** (16-core Xeon):
  - Batch size 1: 20 wafers/second (50ms per wafer)
  - Batch size 8: 100 wafers/second (10ms per wafer amortized)
  - Use case: Fallback when GPUs unavailable, edge deployment
  
- **ONNX + TensorRT Optimization**:
  - FP32: Baseline throughput (200 wafers/sec GPU)
  - FP16: 2× throughput improvement (400 wafers/sec), <0.5% accuracy loss
  - INT8: 3× throughput improvement (600 wafers/sec), <2% accuracy loss (acceptable for early predictions)

**Data Pipeline Throughput**:
- **STDF Ingestion**: 10,000 STDF files/day (continuous from production testers)
- **Wafer Map Generation**: 1,000 wafers/hour (10 parallel workers)
- **Batch Processing**: 5,000 wafers/batch job (overnight processing)
- **Real-Time Streaming**: 10 wafers/second ingested and processed in real-time

**Database Throughput**:
- **PostgreSQL**:
  - Writes: 5,000 predictions/second (INSERT INTO predictions)
  - Reads: 20,000 queries/second (SELECT by wafer_id, cached queries)
  - Concurrent connections: 500 max, 200 typical
  
- **MinIO Object Storage**:
  - Uploads: 500 wafer maps/second (PUT operations)
  - Downloads: 2,000 wafer maps/second (GET operations, CDN-cached)
  - Aggregate throughput: 10 Gbps (saturates network before storage)
  
- **Redis Cache**:
  - Reads: 100,000 requests/second (GET operations, in-memory)
  - Writes: 50,000 requests/second (SET operations with TTL)
  - Cache hit rate target: >80% (reduces database load by 5×)

**API Throughput** (Application Layer):
- **Concurrent Requests**: 1,000 concurrent API requests supported
- **Requests per Second**: 5,000 RPS sustained (across all endpoints)
- **Autoscaling**: Horizontal pod autoscaling (HPA) targets 70% CPU utilization
  - 1 pod: 500 RPS
  - 5 pods: 2,500 RPS
  - 10 pods: 5,000 RPS (max scaling limit)

### 14.3 Resource Usage

**GPU Utilization**:
- **Inference Workload**:
  - Target: 60-80% average GPU utilization (efficient resource usage)
  - Peak: 95% during high-demand periods (acceptable for <10% of time)
  - Idle: <10% (scale down unused GPU instances)
  
- **Training Workload**:
  - Target: 90-95% GPU utilization (maximize training throughput)
  - Mixed Precision (AMP): Reduces memory by 40%, increases utilization by 10-20%
  - GPU Memory: 12GB used of 24GB available (ResNet-18, batch 32), 18GB (ResNet-50, batch 32)

**CPU Utilization**:
- **API Servers**: 40-60% average (leave headroom for traffic spikes)
- **Data Pipeline Workers**: 70-80% average (batch processing jobs)
- **Database**: 30-50% average (well-provisioned for query load)
- **Autoscaling Trigger**: Scale up when CPU >70% for 5 minutes, scale down when <30% for 10 minutes

**Memory Usage**:
- **Inference Service**:
  - Model loading: 500MB (ResNet-18 ONNX), 1.2GB (ResNet-50 ONNX)
  - Per-request overhead: 50MB (preprocessing, batching)
  - Total pod memory: 4GB requested, 8GB limit
  
- **Training Job**:
  - Dataset loading: 2GB (DataLoader cache)
  - Model + optimizer state: 8GB (ResNet-18), 16GB (ResNet-50)
  - Gradient accumulation: +2GB (storing intermediate gradients)
  - Total pod memory: 16GB requested, 32GB limit
  
- **Redis Cache**:
  - Prediction cache: 10GB (1M predictions × 10KB each, TTL=1 hour)
  - Eviction policy: LRU (Least Recently Used)
  - Max memory: 32GB, warning threshold: 24GB (75%)

**Storage Usage**:
- **Wafer Map Images** (MinIO/S3):
  - Per-image size: 100KB compressed PNG (300×300 RGB)
  - Daily ingestion: 1,000 wafers/day × 100KB = 100MB/day
  - Annual growth: 36GB/year
  - Total dataset: 500GB (5 years historical data)
  
- **Model Checkpoints**:
  - Per-checkpoint size: 100MB (ResNet-18 PyTorch), 250MB (ResNet-50 PyTorch)
  - ONNX models: 50MB (ResNet-18), 100MB (ResNet-50)
  - Retention: Keep 10 latest checkpoints per experiment, 200 experiments = 200GB
  
- **Database (PostgreSQL)**:
  - Predictions table: 500 bytes/row, 10M rows = 5GB
  - Wafers table: 1KB/row, 1M rows = 1GB
  - Indexes: 3× data size = 18GB
  - Total: 24GB, grows at 500MB/month
  
- **Logs (OpenSearch)**:
  - Log volume: 50GB/day (structured JSON logs)
  - Retention: 90 days = 4.5TB
  - Compression: 5:1 ratio = 900GB actual storage

**Network Bandwidth**:
- **Ingress**: 
  - STDF uploads: 500MB/hour (peak)
  - API requests: 100Mbps average, 500Mbps peak
  
- **Egress**:
  - Wafer map downloads: 1Gbps average (CDN-cached)
  - Model serving: 200Mbps (ONNX model downloads)
  - Backup/replication: 500Mbps (off-peak hours)
  
- **Total**: 10Gbps network interface (shared), 2Gbps average utilization

**Cost Estimates** (Monthly, AWS Pricing):
- **Compute**:
  - GPU instances (4× A10): 4 × $1.50/hour × 730 hours = $4,380
  - CPU instances (10× m5.2xlarge): 10 × $0.38/hour × 730 hours = $2,774
  - Total compute: $7,154/month
  
- **Storage**:
  - S3 Standard (500GB): $11.50
  - S3 Glacier (2TB archive): $8
  - EBS volumes (2TB SSD): $200
  - Total storage: $219.50/month
  
- **Data Transfer**:
  - Egress (1TB/month): $90
  
- **Managed Services**:
  - RDS PostgreSQL (db.r5.xlarge): $450
  - ElastiCache Redis (cache.r5.large): $180
  - OpenSearch (3× m5.xlarge.search): $900
  - Total managed: $1,530/month
  
- **Grand Total**: ~$9,000/month (~$108K/year) infrastructure cost

---

## 15. Scalability Requirements

### 15.1 Horizontal Scaling

**API Service Scaling**:
- **Kubernetes HPA** (Horizontal Pod Autoscaler):
  - Metric: CPU utilization target 70%, memory target 75%
  - Min replicas: 3 (high availability)
  - Max replicas: 20 (handle traffic spikes)
  - Scale-up: Add 2 pods when CPU >70% for 2 minutes
  - Scale-down: Remove 1 pod when CPU <30% for 10 minutes (conservative to avoid flapping)
  
- **Load Balancing**:
  - NGINX Ingress Controller with round-robin distribution
  - Session affinity: None (stateless API, any pod can handle any request)
  - Health checks: HTTP GET /health every 10 seconds, mark unhealthy after 3 failures
  - Connection draining: 30-second grace period before pod termination

- **Scalability Targets**:
  - Current load: 500 RPS, 5 pods
  - 2× growth (1,000 RPS): 10 pods, linear scaling
  - 5× growth (2,500 RPS): 20 pods (max), consider vertical scaling beyond this
  - 10× growth (5,000 RPS): Multi-region deployment, CDN for static assets

**Inference Service Scaling**:
- **GPU Autoscaling** (Karpenter or Cluster Autoscaler):
  - Monitor GPU utilization across pods (target: 70%)
  - Provision new GPU nodes when existing nodes >80% utilized
  - Deprovision nodes when utilization <20% for 15 minutes
  - Node types: g5.xlarge (1× A10), g5.2xlarge (1× A10G), g5.12xlarge (4× A10G)
  
- **Dynamic Batching**:
  - Collect inference requests for 50ms, batch up to 64 requests
  - Trade-off: 50ms latency increase for 10× throughput improvement
  - Adaptive batching: Reduce wait time to 10ms during low traffic, increase to 100ms during high traffic
  
- **Model Sharding** (for very large models):
  - Split ResNet-50 across 2 GPUs (pipeline parallelism)
  - Layer 1-25 on GPU 0, Layer 26-50 on GPU 1
  - Use case: Future models >25M parameters (ViT, EfficientNet-L2)

**Database Scaling**:
- **PostgreSQL Read Replicas**:
  - 1 primary (writes), 3 read replicas (reads)
  - Connection pooling: PgBouncer (500 client connections → 100 DB connections)
  - Read-heavy queries (predictions, wafer lookups) routed to replicas
  - Write queries (new predictions, labels) routed to primary
  
- **Partitioning**:
  - Predictions table partitioned by month (PARTITION BY RANGE (timestamp))
  - Automatic partition creation: create_monthly_partitions() cron job
  - Old partition archival: Move partitions >1 year to archive table, then S3
  
- **Connection Limits**:
  - Primary: max_connections=500, reserved 100 for admin
  - Replicas: max_connections=1000 (read-only, less contention)

**Storage Scaling**:
- **MinIO/S3 Object Storage**:
  - Horizontally scalable by design (distributed object storage)
  - Add storage nodes as needed (16TB → 32TB → 64TB)
  - Multi-region replication for disaster recovery (primary: us-east-1, backup: eu-west-1)
  
- **Redis Cache Scaling**:
  - Redis Cluster mode: 6 shards (3 primary, 3 replica)
  - Sharding by key hash (wafer_id, prediction_id)
  - Vertical scaling: Upgrade from cache.r5.large (13GB) to cache.r5.xlarge (26GB) as cache size grows

### 15.2 Vertical Scaling

**Compute Vertical Scaling**:
- **GPU Upgrades**:
  - Phase 1: NVIDIA A10 (24GB, $1.50/hour, 31.2 TFLOPS FP32)
  - Phase 2: NVIDIA A100 (40GB, $3.20/hour, 156 TFLOPS FP32) - 5× faster training
  - Phase 3: NVIDIA H100 (80GB, $8.00/hour, 756 TFLOPS FP32) - for future large models (ViT-Huge)
  
- **CPU Upgrades**:
  - Phase 1: m5.2xlarge (8 vCPU, 32GB RAM) - API servers
  - Phase 2: m5.4xlarge (16 vCPU, 64GB RAM) - handle 2× traffic
  - Phase 3: c5.12xlarge (48 vCPU, 96GB RAM) - CPU-intensive batch processing

**Database Vertical Scaling**:
- **PostgreSQL Instance Sizes**:
  - Phase 1: db.r5.xlarge (4 vCPU, 32GB RAM) - 10M predictions
  - Phase 2: db.r5.2xlarge (8 vCPU, 64GB RAM) - 50M predictions
  - Phase 3: db.r5.4xlarge (16 vCPU, 128GB RAM) - 100M predictions
  - Upgrade window: Sunday 2-4 AM (minimal traffic), <30 min downtime with RDS Multi-AZ

**Memory Vertical Scaling**:
- **Inference Pods**:
  - Current: 4GB request, 8GB limit (ResNet-18)
  - ResNet-50: 8GB request, 16GB limit
  - Future ViT models: 16GB request, 32GB limit
  
- **Training Jobs**:
  - Current: 16GB request, 32GB limit (ResNet-18, batch 32)
  - Large batch: 32GB request, 64GB limit (batch 128 for faster convergence)
  - Multi-GPU: 64GB request, 128GB limit (4× A100 GPUs)

**Storage Vertical Scaling**:
- **EBS Volumes** (PostgreSQL, logs):
  - Phase 1: 500GB gp3 (3,000 IOPS, 125 MB/s)
  - Phase 2: 2TB gp3 (12,000 IOPS, 500 MB/s)
  - Phase 3: 5TB io2 (64,000 IOPS, 1,000 MB/s) - for high write workloads
  
- **MinIO Storage**:
  - Phase 1: 16TB (4× 4TB drives per node, 4 nodes = 16TB raw)
  - Phase 2: 64TB (4× 4TB drives per node, 16 nodes = 64TB raw)
  - Erasure coding: 8+4 (8 data, 4 parity) = ~66% usable capacity

### 15.3 Load Handling

**Traffic Spike Handling**:
- **Autoscaling Response Time**:
  - Pod creation: 30-60 seconds (pull container image, start application)
  - Node provisioning: 3-5 minutes (EC2 instance launch, join cluster)
  - Pre-warming: Keep 20% excess capacity during business hours (8 AM - 6 PM)
  
- **Request Queueing**:
  - API Gateway queue: 10,000 requests max (overflow returns 503 Service Unavailable)
  - Batch job queue: Celery with Redis backend, unlimited queue size
  - Priority queuing: HIGH priority jobs bypass queue, NORMAL/LOW queued in order

**Burst Capacity**:
- **API Burst**:
  - Normal: 500 RPS, 5 pods
  - Burst (2× traffic): 1,000 RPS, scale to 10 pods in 2 minutes
  - Extreme burst (5× traffic): 2,500 RPS, scale to 20 pods in 5 minutes + activate read replicas
  
- **GPU Inference Burst**:
  - Normal: 1,000 wafers/hour, 2 GPU pods
  - Burst: 5,000 wafers/hour, scale to 10 GPU pods (provision 8 new g5.xlarge nodes)
  - Fallback: Overflow to CPU inference (slower but prevents request failures)

**Multi-Region Deployment** (Future):
- **Geo-Distribution**:
  - Primary region: US-East (manufacturing sites in Texas, Arizona)
  - Secondary region: EU-West (manufacturing sites in Germany, Ireland)
  - Asia-Pacific region: Singapore (manufacturing sites in Taiwan, Korea, Japan)
  
- **Data Residency**:
  - Wafer data stored in region where manufactured (GDPR, data sovereignty)
  - Models replicated globally (no PII in models)
  - Cross-region prediction: Route to nearest region with latency <100ms
  
- **Failover Strategy**:
  - Active-Active: Both regions serve traffic (Route 53 latency-based routing)
  - Failover: If primary region unavailable, redirect 100% traffic to secondary within 5 minutes
  - Data sync: Bidirectional replication with conflict resolution (last-write-wins)

**Caching Strategy for Scale**:
- **CDN (CloudFront)**:
  - Cache static assets: wafer map images (300×300 PNG), Grad-CAM heatmaps
  - TTL: 7 days (wafer maps immutable after creation)
  - Cache hit rate target: >90% (10× reduction in origin requests)
  
- **Application Cache (Redis)**:
  - Hot predictions: Cache recent predictions (TTL=1 hour), 80% hit rate
  - Model metadata: Cache model versions, configs (TTL=1 day), 95% hit rate
  - User sessions: Store JWT tokens, user preferences (TTL=session duration)
  
- **Database Query Cache**:
  - Materialized views: Pre-compute aggregations (daily yield trends, defect distributions)
  - Refresh schedule: Every 1 hour during business hours, every 6 hours off-hours
  - Example: `CREATE MATERIALIZED VIEW daily_yield_summary AS SELECT ...`

**Database Connection Pooling**:
- **PgBouncer**:
  - Pool mode: Transaction (connection returned to pool after each transaction)
  - Pool size: 100 connections per database
  - Client connections: 500 max (5× multiplexing ratio)
  - Idle timeout: 300 seconds (release unused connections)
  
- **Application-Level Pooling**:
  - SQLAlchemy pool: pool_size=20, max_overflow=40 (60 max connections per API pod)
  - Connection lifetime: 1 hour (prevent stale connections)
  - Pre-ping: Test connection health before use (detect broken connections)

**Graceful Degradation**:
- **Fallback Mechanisms**:
  - GPU unavailable → CPU inference (10× slower but functional)
  - Database overload → Serve cached predictions, queue writes
  - Model inference failure → Rule-based estimate (e.g., PASS/FAIL ratio as yield estimate)
  - CDN/S3 down → Generate wafer maps on-the-fly (slower but functional)
  
- **Circuit Breaker Pattern**:
  - Detect repeated failures (e.g., 50% of inference requests fail in 1 minute)
  - Open circuit: Stop sending requests to failing service, return cached/fallback response
  - Half-open: After 30 seconds, test with 1 request, close circuit if successful
  - Prevents cascading failures and reduces load on struggling services

---

## 16. Testing Strategy

### 16.1 Unit Testing

**Code Coverage Targets**:
- **Core Modules**: >90% coverage (data ingestion, model training, inference, API endpoints)
- **Utilities**: >80% coverage (helper functions, preprocessing)
- **UI Components**: >70% coverage (React components, interactions)
- **Overall Target**: >85% code coverage across project

**Testing Framework**:
- **Python Backend**: pytest 8.0+, pytest-cov for coverage, pytest-mock for mocking
- **JavaScript Frontend**: Jest 29+, React Testing Library, MSW (Mock Service Worker) for API mocking

**Test Categories**:

1. **Data Processing Tests**:
   ```python
   # tests/unit/test_stdf_parser.py
   def test_parse_stdf_file():
       """Test STDF parsing extracts correct die coordinates and bins"""
       stdf_path = "tests/fixtures/sample_wafer.stdf"
       result = parse_stdf(stdf_path)
       
       assert result["wafer_id"] == "W12345-001"
       assert result["die_count"] == 5000
       assert result["pass_count"] == 4350
       assert len(result["die_coordinates"]) == 5000
       assert result["die_coordinates"][0] == {"x": 0, "y": 0, "bin": 1}  # PASS
   
   def test_generate_wafer_map():
       """Test wafer map generation creates valid 300×300 RGB image"""
       die_data = [{"x": i, "y": j, "bin": 1 if i+j < 50 else 5} 
                   for i in range(100) for j in range(50)]
       
       wafer_map = generate_wafer_map(die_data, size=(300, 300))
       
       assert wafer_map.shape == (300, 300, 3)  # RGB image
       assert wafer_map.dtype == np.uint8
       assert np.all(wafer_map >= 0) and np.all(wafer_map <= 255)
       # Check green pixels (PASS) exist
       assert np.any((wafer_map == [0, 255, 0]).all(axis=2))
   ```

2. **Model Tests**:
   ```python
   # tests/unit/test_model.py
   def test_resnet_forward_pass():
       """Test ResNet model forward pass produces valid output shape"""
       model = WaferYieldPredictor(num_classes=8, pretrained=False)
       input_tensor = torch.randn(4, 3, 224, 224)  # batch=4
       
       output = model(input_tensor)
       
       assert output.shape == (4, 8)  # 4 samples, 8 classes
       assert torch.all(torch.isfinite(output))  # No NaN/Inf
   
   def test_freeze_backbone():
       """Test freeze_backbone disables gradient computation for backbone"""
       model = WaferYieldPredictor(num_classes=8)
       model.freeze_backbone()
       
       # Check backbone layers frozen
       for name, param in model.named_parameters():
           if "fc" not in name:  # Not final layer
               assert not param.requires_grad
       
       # Check final layer trainable
       assert model.resnet.fc.weight.requires_grad
   ```

3. **API Tests**:
   ```python
   # tests/unit/test_api.py
   @pytest.fixture
   def client():
       """FastAPI test client"""
       from api.main import app
       return TestClient(app)
   
   def test_predict_endpoint(client, mocker):
       """Test /predict endpoint returns valid prediction"""
       # Mock inference service
       mock_predict = mocker.patch("api.routes.inference_service.predict")
       mock_predict.return_value = {
           "yield": 87.3, 
           "defect_class": "EdgeEffect", 
           "confidence": 0.92
       }
       
       response = client.post(
           "/api/v1/predict",
           files={"wafer_map_image": ("test.png", open("tests/fixtures/wafer.png", "rb"))}
       )
       
       assert response.status_code == 200
       data = response.json()
       assert data["prediction"]["yield"] == 87.3
       assert data["prediction"]["defect_class"] == "EdgeEffect"
   ```

4. **Preprocessing Tests**:
   ```python
   # tests/unit/test_preprocessing.py
   def test_image_normalization():
       """Test image normalization uses ImageNet statistics"""
       image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
       
       normalized = preprocess_image(image)
       
       # Check shape and type
       assert normalized.shape == (3, 224, 224)  # CHW format
       assert normalized.dtype == np.float32
       
       # Check normalization (mean ≈ 0, std ≈ 1 for ImageNet stats)
       assert -3 < normalized.mean() < 3
       assert 0.5 < normalized.std() < 2
   ```

### 16.2 Integration Testing

**End-to-End Workflows**:

1. **STDF Upload → Prediction Workflow**:
   ```python
   # tests/integration/test_e2e_prediction.py
   def test_stdf_upload_to_prediction(test_db, test_storage):
       """Test complete workflow: STDF upload → wafer map gen → inference → DB storage"""
       # 1. Upload STDF file
       stdf_path = "tests/fixtures/sample_wafer.stdf"
       wafer_id = upload_stdf(stdf_path)
       
       # 2. Verify wafer map generated and stored in MinIO
       wafer_map_url = get_wafer_map_url(wafer_id)
       assert wafer_map_url.startswith("s3://")
       assert storage_client.exists(wafer_map_url)
       
       # 3. Run prediction
       prediction = predict_yield(wafer_id)
       assert "yield" in prediction
       assert "defect_class" in prediction
       assert 0 <= prediction["yield"] <= 100
       
       # 4. Verify prediction stored in PostgreSQL
       db_prediction = test_db.query(Prediction).filter_by(wafer_id=wafer_id).first()
       assert db_prediction is not None
       assert db_prediction.yield_pred == prediction["yield"]
   ```

2. **Model Training → Deployment Workflow**:
   ```python
   # tests/integration/test_training_deployment.py
   def test_model_training_to_deployment(test_mlflow):
       """Test training → MLflow logging → ONNX export → model registry"""
       # 1. Train model (small dataset for speed)
       config = {
           "architecture": "resnet18",
           "epochs": 2,
           "batch_size": 16
       }
       run_id = train_model(config, dataset="tests/fixtures/mini_dataset")
       
       # 2. Verify MLflow run logged
       run = test_mlflow.get_run(run_id)
       assert run.data.params["architecture"] == "resnet18"
       assert "val_accuracy" in run.data.metrics
       
       # 3. Export to ONNX
       onnx_path = export_to_onnx(run_id)
       assert os.path.exists(onnx_path)
       assert onnx_path.endswith(".onnx")
       
       # 4. Register model
       model_version = register_model(run_id, stage="STAGING")
       assert model_version.stage == "STAGING"
   ```

3. **Database Integration Tests**:
   ```python
   # tests/integration/test_database.py
   def test_prediction_query_performance(test_db):
       """Test prediction queries meet performance SLO (<100ms)"""
       # Insert 10,000 test predictions
       predictions = [
           Prediction(wafer_id=f"W{i:05d}", yield_pred=random.uniform(70, 95))
           for i in range(10000)
       ]
       test_db.bulk_save_objects(predictions)
       test_db.commit()
       
       # Query by wafer_id (indexed)
       start = time.time()
       result = test_db.query(Prediction).filter_by(wafer_id="W05000").first()
       duration_ms = (time.time() - start) * 1000
       
       assert result is not None
       assert duration_ms < 100  # SLO: <100ms
   ```

**API Integration Tests**:
```python
# tests/integration/test_api_integration.py
def test_batch_prediction_job(client, test_db):
    """Test batch prediction creates job, processes wafers, returns results"""
    # 1. Submit batch job
    response = client.post(
        "/api/v1/predict/batch",
        json={"wafer_ids": ["W001", "W002", "W003"]}
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    
    # 2. Poll job status
    max_wait = 60  # 60 seconds timeout
    start = time.time()
    while time.time() - start < max_wait:
        status_response = client.get(f"/api/v1/jobs/{job_id}")
        status = status_response.json()["status"]
        if status == "COMPLETED":
            break
        time.sleep(1)
    
    assert status == "COMPLETED"
    
    # 3. Verify results stored in database
    predictions = test_db.query(Prediction).filter(
        Prediction.wafer_id.in_(["W001", "W002", "W003"])
    ).all()
    assert len(predictions) == 3
```

### 16.3 Performance Testing

**Load Testing** (Locust):

```python
# tests/performance/locustfile.py
from locust import HttpUser, task, between

class YieldPredictorUser(HttpUser):
    wait_time = between(1, 3)  # 1-3 seconds between requests
    
    def on_start(self):
        """Login and get JWT token"""
        response = self.client.post("/api/v1/auth/login", json={
            "username": "test_user",
            "password": "test_password"
        })
        self.token = response.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task(3)  # 60% of requests
    def predict_yield(self):
        """Simulate prediction request"""
        with open("tests/fixtures/wafer.png", "rb") as f:
            self.client.post(
                "/api/v1/predict",
                headers=self.headers,
                files={"wafer_map_image": f}
            )
    
    @task(2)  # 40% of requests
    def similarity_search(self):
        """Simulate similarity search"""
        self.client.post(
            "/api/v1/similarity-search",
            headers=self.headers,
            json={"wafer_id": "W12345-001", "top_k": 10}
        )
```

**Load Test Scenarios**:
1. **Baseline Load**: 100 users, 500 RPS, 10 minutes (current production load)
2. **Peak Load**: 200 users, 1,000 RPS, 10 minutes (2× growth scenario)
3. **Stress Test**: 500 users, 2,500 RPS, 10 minutes (find breaking point)
4. **Spike Test**: 100 → 500 → 100 users over 5 minutes (test autoscaling)
5. **Soak Test**: 100 users, 24 hours (detect memory leaks, performance degradation)

**Performance Metrics**:
- **Response Time**: p50, p95, p99 latency for each endpoint
- **Throughput**: Requests per second sustained
- **Error Rate**: % of requests returning 4xx or 5xx
- **Resource Usage**: CPU, memory, GPU utilization during load

**Inference Performance Tests**:
```python
# tests/performance/test_inference_perf.py
import torch
import time

def test_gpu_inference_throughput():
    """Measure GPU inference throughput (wafers/second)"""
    model = load_onnx_model("models/resnet18-v1.2.onnx", device="cuda")
    batch_sizes = [1, 8, 16, 32, 64]
    
    for batch_size in batch_sizes:
        inputs = torch.randn(batch_size, 3, 224, 224).cuda()
        
        # Warmup
        for _ in range(10):
            model(inputs)
        
        # Measure
        start = time.time()
        num_iterations = 100
        for _ in range(num_iterations):
            model(inputs)
        torch.cuda.synchronize()
        duration = time.time() - start
        
        throughput = (num_iterations * batch_size) / duration
        latency_ms = (duration / (num_iterations * batch_size)) * 1000
        
        print(f"Batch size {batch_size}: {throughput:.1f} wafers/sec, {latency_ms:.2f} ms/wafer")
        
        # Assert throughput meets target
        if batch_size == 32:
            assert throughput > 1500  # Target: >1,500 wafers/sec at batch=32
```

### 16.4 Security Testing

**SAST (Static Application Security Testing)**:
- **Tools**: SonarQube, Bandit (Python), ESLint security plugins (JavaScript)
- **Checks**: SQL injection, XSS, hardcoded secrets, insecure randomness, path traversal
- **CI/CD Integration**: Fail build if critical vulnerabilities detected
- **Baseline**: Zero high/critical issues, <10 medium issues

**DAST (Dynamic Application Security Testing)**:
- **Tools**: OWASP ZAP, Burp Suite
- **Tests**: 
  - Authentication bypass attempts
  - Authorization testing (privilege escalation)
  - SQL injection, XSS, CSRF
  - Broken authentication, session management
  
- **Scan Frequency**: Weekly on staging environment
- **False Positive Review**: Security team reviews findings, marks false positives

**Dependency Vulnerability Scanning**:
```yaml
# .github/workflows/security-scan.yml
name: Security Scan

on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight
  pull_request:

jobs:
  dependency-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Snyk
        uses: snyk/actions/python@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high
      
      - name: Run Trivy (container scan)
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'myregistry/yield-predictor:latest'
          format: 'sarif'
          output: 'trivy-results.sarif'
```

**Penetration Testing**:
- **Frequency**: Annual pen test by third-party security firm
- **Scope**: External attack surface (API, web UI), internal network segmentation
- **Remediation SLA**: Critical findings <7 days, high <30 days, medium <90 days
- **Retest**: Verify fixes within 30 days of initial pen test

**Security Regression Tests**:
```python
# tests/security/test_auth.py
def test_unauthenticated_request_blocked(client):
    """Verify API rejects requests without valid JWT"""
    response = client.get("/api/v1/models")
    assert response.status_code == 401
    assert "unauthorized" in response.json()["detail"].lower()

def test_expired_token_rejected(client):
    """Verify expired JWT tokens are rejected"""
    expired_token = create_jwt_token(user_id="test", exp=datetime.now() - timedelta(hours=1))
    response = client.get(
        "/api/v1/models",
        headers={"Authorization": f"Bearer {expired_token}"}
    )
    assert response.status_code == 401

def test_insufficient_permissions(client, model_user_token):
    """Verify model_user cannot promote models (requires model_admin role)"""
    response = client.post(
        "/api/v1/models/resnet18-v1.2/promote",
        headers={"Authorization": f"Bearer {model_user_token}"},
        json={"target_stage": "PRODUCTION"}
    )
    assert response.status_code == 403
    assert "insufficient permissions" in response.json()["detail"].lower()
```

---

## 17. Deployment Strategy

### 17.1 Deployment Pipeline

**CI/CD Workflow** (GitHub Actions):

```yaml
# .github/workflows/deploy.yml
name: Build and Deploy

on:
  push:
    branches: [main, staging]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run linters
        run: |
          black --check src/
          isort --check src/
          flake8 src/
          mypy src/
      
      - name: Run unit tests
        run: pytest tests/unit --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
  
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ github.repository }}/api:${{ github.sha }}
            ghcr.io/${{ github.repository }}/api:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
  
  deploy-staging:
    needs: build
    if: github.ref == 'refs/heads/staging'
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up kubectl
        uses: azure/setup-kubectl@v4
      
      - name: Deploy to staging with Helm
        run: |
          helm upgrade --install yield-predictor ./helm \
            --namespace staging \
            --set image.tag=${{ github.sha }} \
            --set env=staging \
            --wait
      
      - name: Run integration tests
        run: |
          pytest tests/integration --base-url=https://staging.example.com
  
  deploy-production:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to production (canary)
        run: |
          helm upgrade --install yield-predictor ./helm \
            --namespace production \
            --set image.tag=${{ github.sha }} \
            --set env=production \
            --set canary.enabled=true \
            --set canary.weight=10 \
            --wait
      
      - name: Monitor canary metrics
        run: |
          ./scripts/monitor-canary.sh --duration=15m --threshold=5
      
      - name: Promote canary to full deployment
        run: |
          helm upgrade yield-predictor ./helm \
            --namespace production \
            --set image.tag=${{ github.sha }} \
            --set canary.enabled=false \
            --wait
```

**Docker Build Optimization**:
```dockerfile
# Dockerfile (multi-stage build)
FROM python:3.11-slim AS base
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Stage 1: Build dependencies
FROM base AS builder
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Final image
FROM base
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY src/ ./src/
COPY models/ ./models/

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 17.2 Environments

**Environment Configuration**:

| Environment | Purpose | Data | Users | Autoscaling | Deployment |
|------------|---------|------|-------|-------------|------------|
| **Development** | Local dev | Synthetic | Developers | No | Manual |
| **Staging** | Pre-prod testing | Anonymized prod data | QA, ML Engineers | Yes (limited) | Auto (staging branch) |
| **Production** | Live system | Real prod data | All users | Yes (full) | Auto (main branch) |
| **DR (Disaster Recovery)** | Failover | Replicated prod data | None (standby) | Yes (activated on failover) | Manual trigger |

**Environment-Specific Configs**:
```python
# config/settings.py
class Config:
    """Base configuration"""
    DEBUG = False
    TESTING = False
    DATABASE_POOL_SIZE = 20
    REDIS_TTL = 3600  # 1 hour

class DevelopmentConfig(Config):
    DEBUG = True
    DATABASE_URL = "postgresql://localhost/yield_predictor_dev"
    REDIS_URL = "redis://localhost:6379/0"
    MODEL_PATH = "models/dev"

class StagingConfig(Config):
    DATABASE_URL = os.getenv("DATABASE_URL")  # from secrets
    REDIS_URL = os.getenv("REDIS_URL")
    MODEL_PATH = "s3://models-staging"
    LOG_LEVEL = "INFO"

class ProductionConfig(Config):
    DATABASE_URL = os.getenv("DATABASE_URL")
    REDIS_URL = os.getenv("REDIS_URL")
    MODEL_PATH = "s3://models-production"
    LOG_LEVEL = "WARNING"
    DATABASE_POOL_SIZE = 50  # higher pool for prod
```

### 17.3 Rollout Plan

**Phase 1: Offline Validation** (Weeks 1-2):
- Deploy to staging environment
- Run batch predictions on historical data (10,000 wafers)
- Compare predictions to actual yield (calculate MAE, RMSE, accuracy)
- Validate Grad-CAM heatmaps with yield engineer review (sample 100 wafers)
- Performance testing: load test with 2× production traffic
- **Success Criteria**: Accuracy >92%, latency <200ms p95, zero critical bugs

**Phase 2: Shadow Mode** (Weeks 3-4):
- Deploy to production (shadow mode: predictions not visible to users)
- Run predictions in parallel with existing system (if any)
- Log all predictions to database for later analysis
- Compare ML predictions vs. actual yield (continuous validation)
- Monitor for data drift, concept drift (model accuracy degradation)
- **Success Criteria**: Accuracy maintained >90%, no production incidents, 100% uptime

**Phase 3: Limited Beta** (Weeks 5-8):
- Enable UI for 10 pilot users (yield engineers, test engineers)
- Predictions visible but marked "Beta - For Review Only"
- Collect user feedback via in-app surveys (thumbs up/down on predictions)
- Weekly meetings with pilot users to discuss pain points, feature requests
- Iterate on UI/UX based on feedback
- **Success Criteria**: >4.0/5.0 user satisfaction, 80% of predictions reviewed, <5 bugs reported

**Phase 4: Assisted Prediction** (Weeks 9-12):
- Roll out to 50% of users (A/B test)
- Predictions provided as recommendations, final decision by engineer
- Track adoption: % of engineers using predictions in daily work
- Measure impact: time saved on manual wafer map review
- Gradual rollout: 50% → 75% → 100% over 4 weeks
- **Success Criteria**: 60% adoption rate, 50% reduction in manual review time, <2% error rate

**Phase 5: Automated Termination** (Weeks 13-16):
- Enable adaptive test termination for high-confidence predictions (confidence >95%)
- Conservative thresholds: only terminate when yield prediction >85% or <20%
- Human-in-the-loop for edge cases: flag low-confidence predictions for manual review
- Monitor DPPM impact: ensure no increase in field failures
- **Success Criteria**: 20% test time reduction, DPPM increase <1%, no safety incidents

**Phase 6: Full Production** (Week 17+):
- 100% rollout to all users and products
- Remove "Beta" labels, predictions are official
- Continuous improvement: monthly model retraining, quarterly feature releases
- **Success Criteria**: $3M+ annual cost savings achieved, >4.2/5.0 user satisfaction

### 17.4 Rollback Procedures

**Rollback Triggers**:
- Prediction accuracy drops >5% below baseline (e.g., 92% → 87%)
- API error rate >5% for >10 minutes
- p95 latency >500ms (2.5× SLO)
- Critical security vulnerability discovered (CVSS >9.0)
- User-reported critical bug affecting production decisions
- DPPM increase >2% (quality impact)

**Rollback Process** (GitOps with ArgoCD):
1. **Detect Issue**: Automated alerts trigger (Prometheus, Grafana, PagerDuty)
2. **Decision**: On-call engineer assesses severity (rollback vs. hotfix)
3. **Execute Rollback**: 
   ```bash
   # Rollback to previous Helm release
   helm rollback yield-predictor -n production
   
   # Or rollback ArgoCD application
   argocd app rollback yield-predictor --revision <previous-revision>
   ```
4. **Verify**: Check health metrics, run smoke tests
5. **Communicate**: Notify users via Slack/email about temporary rollback
6. **Root Cause Analysis**: Post-mortem within 24 hours, action items assigned
7. **Fix Forward**: Implement fix, test in staging, re-deploy

**Model Rollback** (Model Registry):
```python
# scripts/rollback-model.py
def rollback_model(previous_version="resnet18-v1.1"):
    """Rollback to previous production model"""
    # 1. Fetch previous model from registry
    model = mlflow.pyfunc.load_model(f"models:/{previous_version}/Production")
    
    # 2. Update model serving config
    update_serving_config(model_path=model.model_uri)
    
    # 3. Clear Redis cache (force re-inference with old model)
    redis_client.flushdb()
    
    # 4. Update model_version in database
    db.execute(f"UPDATE system_config SET production_model = '{previous_version}'")
    
    # 5. Log rollback event
    log_event("MODEL_ROLLBACK", metadata={
        "from_version": current_version,
        "to_version": previous_version,
        "reason": "Accuracy degradation detected"
    })
```

**Canary Rollback**:
- If canary deployment (10% traffic) shows issues, abort promotion automatically
- ArgoCD analysis: compare canary metrics to baseline for 15 minutes
- Metrics: error_rate, latency_p95, prediction_accuracy
- Rollback if any metric degrades >10% compared to baseline

**Blue-Green Deployment** (Zero-Downtime Rollback):
- Maintain two identical environments: Blue (current production), Green (new version)
- Deploy to Green, run smoke tests, gradually shift traffic (10% → 50% → 100%)
- If issues detected, instant rollback by shifting traffic back to Blue
- Green becomes new Blue after successful deployment, old Blue kept for 24 hours then decommissioned

---

## 18. Monitoring & Observability

### 18.1 Metrics

**Application Metrics** (Prometheus):

1. **API Metrics**:
   ```python
   # Latency histogram
   api_latency = Histogram(
       'api_request_duration_seconds',
       'API request latency',
       ['endpoint', 'method', 'status_code'],
       buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
   )
   
   # Request counter
   api_requests_total = Counter(
       'api_requests_total',
       'Total API requests',
       ['endpoint', 'method', 'status_code']
   )
   
   # Active requests gauge
   api_requests_active = Gauge(
       'api_requests_active',
       'Number of active API requests'
   )
   ```
   
   **Tracked Metrics**:
   - Request rate: requests/second per endpoint
   - Latency: p50, p95, p99, p999 per endpoint
   - Error rate: % of 4xx and 5xx responses
   - Active connections: concurrent requests being processed

2. **Inference Metrics**:
   ```python
   # Inference latency
   inference_latency = Histogram(
       'inference_duration_seconds',
       'Model inference latency',
       ['model_version', 'device'],
       buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
   )
   
   # Prediction confidence
   prediction_confidence = Histogram(
       'prediction_confidence',
       'Model prediction confidence score',
       ['model_version', 'defect_class'],
       buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
   )
   
   # Predictions counter
   predictions_total = Counter(
       'predictions_total',
       'Total predictions made',
       ['model_version', 'defect_class']
   )
   ```
   
   **Tracked Metrics**:
   - Inference throughput: predictions/second
   - Inference latency: p50, p95, p99 (GPU vs. CPU)
   - Prediction distribution: count per defect class
   - Confidence distribution: histogram of confidence scores
   - Batch size: average batch size for dynamic batching

3. **Model Performance Metrics**:
   ```python
   # Prediction accuracy (requires ground truth)
   prediction_accuracy = Gauge(
       'prediction_accuracy',
       'Model prediction accuracy (daily)',
       ['model_version', 'product_id']
   )
   
   # MAE (Mean Absolute Error)
   prediction_mae = Gauge(
       'prediction_mae',
       'Mean Absolute Error for yield predictions',
       ['model_version']
   )
   
   # Data drift indicator
   data_drift_score = Gauge(
       'data_drift_score',
       'Data drift score (KL divergence)',
       ['feature_name']
   )
   ```
   
   **Tracked Metrics**:
   - Daily accuracy: compare predictions vs. actual yield
   - MAE, RMSE: regression error metrics
   - Confusion matrix: per-class precision/recall
   - Data drift: KL divergence between training and production data distributions
   - Concept drift: accuracy trend over time (7-day, 30-day moving average)

4. **GPU Metrics**:
   ```python
   # GPU utilization
   gpu_utilization = Gauge(
       'gpu_utilization_percent',
       'GPU utilization percentage',
       ['gpu_id', 'node']
   )
   
   # GPU memory
   gpu_memory_used = Gauge(
       'gpu_memory_used_bytes',
       'GPU memory used',
       ['gpu_id', 'node']
   )
   
   # GPU temperature
   gpu_temperature = Gauge(
       'gpu_temperature_celsius',
       'GPU temperature',
       ['gpu_id', 'node']
   )
   ```
   
   **Tracked Metrics**:
   - GPU utilization: % (target: 60-80%)
   - GPU memory: used/total (GB)
   - GPU temperature: °C (alert if >85°C)
   - GPU power draw: watts
   - CUDA errors: count (OOM, illegal memory access)

5. **Cache Metrics**:
   ```python
   # Cache hits/misses
   cache_hits_total = Counter('cache_hits_total', 'Cache hits', ['cache_type'])
   cache_misses_total = Counter('cache_misses_total', 'Cache misses', ['cache_type'])
   
   # Cache hit rate gauge
   cache_hit_rate = Gauge('cache_hit_rate', 'Cache hit rate', ['cache_type'])
   ```
   
   **Tracked Metrics**:
   - Hit rate: % (target: >80% for predictions)
   - Memory usage: MB/GB
   - Eviction rate: keys evicted/second
   - Key count: total keys in cache

6. **Database Metrics**:
   - Connection pool: active/idle/waiting connections
   - Query latency: p50, p95, p99 per query type
   - Slow queries: queries >100ms
   - Deadlocks: count per hour
   - Replication lag: seconds (for read replicas)

### 18.2 Logging

**Structured Logging** (JSON Format):

```python
# src/core/logging_config.py
import logging
import json
from datetime import datetime
from contextvars import ContextVar

# Correlation ID for request tracing
correlation_id_var: ContextVar[str] = ContextVar('correlation_id', default='')

class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging"""
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add custom fields from extra
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'lineno', 'module', 'msecs', 'message',
                          'pathname', 'process', 'processName', 'relativeCreated',
                          'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info']:
                log_data[key] = value
        
        return json.dumps(log_data)

# Configure logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.handlers[0].setFormatter(JSONFormatter())
```

**Log Levels**:
- **DEBUG**: Detailed debugging info (disabled in production)
- **INFO**: General informational messages (API requests, predictions made)
- **WARNING**: Warning messages (cache miss, slow query, low confidence prediction)
- **ERROR**: Error messages (inference failure, database timeout, validation error)
- **CRITICAL**: Critical failures (GPU failure, database connection lost, model load failed)

**Log Examples**:

1. **API Request Log**:
   ```json
   {
     "timestamp": "2025-12-04T10:30:45.123Z",
     "level": "INFO",
     "logger": "api.main",
     "message": "API request received",
     "correlation_id": "abc123-def456",
     "method": "POST",
     "endpoint": "/api/v1/predict",
     "user_id": "user@example.com",
     "wafer_id": "W12345-001",
     "status_code": 200,
     "latency_ms": 187
   }
   ```

2. **Inference Log**:
   ```json
   {
     "timestamp": "2025-12-04T10:30:45.250Z",
     "level": "INFO",
     "logger": "inference.service",
     "message": "Prediction completed",
     "correlation_id": "abc123-def456",
     "wafer_id": "W12345-001",
     "model_version": "resnet18-v1.2",
     "yield_pred": 87.3,
     "defect_class": "EdgeEffect",
     "confidence": 0.92,
     "inference_time_ms": 152,
     "device": "cuda:0"
   }
   ```

3. **Error Log**:
   ```json
   {
     "timestamp": "2025-12-04T10:31:22.456Z",
     "level": "ERROR",
     "logger": "inference.service",
     "message": "Model inference failed",
     "correlation_id": "xyz789-uvw012",
     "wafer_id": "W12346-005",
     "model_version": "resnet18-v1.2",
     "error_type": "RuntimeError",
     "error_message": "CUDA out of memory",
     "exception": "Traceback (most recent call last):\n  File ...",
     "retry_count": 2
   }
   ```

4. **Model Performance Log** (Daily):
   ```json
   {
     "timestamp": "2025-12-04T23:59:59.999Z",
     "level": "INFO",
     "logger": "monitoring.daily_metrics",
     "message": "Daily model performance summary",
     "model_version": "resnet18-v1.2",
     "date": "2025-12-04",
     "predictions_count": 12450,
     "accuracy": 0.9245,
     "mae": 2.87,
     "rmse": 4.12,
     "avg_confidence": 0.89,
     "defect_distribution": {
       "EdgeEffect": 3200,
       "CenterCluster": 1800,
       "Normal": 5400,
       "RingPattern": 450,
       "Other": 1600
     }
   }
   ```

**Log Aggregation** (OpenSearch/ELK):
- **Ingestion**: Filebeat/Fluentd collects logs from all pods → Logstash → OpenSearch
- **Indexing**: Daily indices (logs-2025-12-04, logs-2025-12-05)
- **Retention**: 90 days in hot storage, 1 year in warm storage, then delete
- **Queries**: Kibana dashboards for log search, filtering, visualization

### 18.3 Alerting

**Alerting Rules** (Prometheus Alertmanager):

1. **High Error Rate**:
   ```yaml
   - alert: HighErrorRate
     expr: |
       (sum(rate(api_requests_total{status_code=~"5.."}[5m]))
        / sum(rate(api_requests_total[5m]))) > 0.05
     for: 5m
     labels:
       severity: critical
     annotations:
       summary: "High API error rate (> 5%)"
       description: "Error rate is {{ $value | humanizePercentage }} for last 5 minutes"
   ```

2. **High Latency**:
   ```yaml
   - alert: HighLatency
     expr: |
       histogram_quantile(0.95, 
         sum(rate(api_latency_bucket[5m])) by (endpoint, le)
       ) > 0.5
     for: 10m
     labels:
       severity: warning
     annotations:
       summary: "High API latency (p95 > 500ms)"
       description: "p95 latency is {{ $value }}s for endpoint {{ $labels.endpoint }}"
   ```

3. **Model Accuracy Degradation**:
   ```yaml
   - alert: ModelAccuracyDegraded
     expr: prediction_accuracy < 0.87  # 5% below 92% baseline
     for: 1h
     labels:
       severity: critical
     annotations:
       summary: "Model accuracy degraded (< 87%)"
       description: "Model {{ $labels.model_version }} accuracy is {{ $value | humanizePercentage }}"
   ```

4. **GPU Failure**:
   ```yaml
   - alert: GPUUnavailable
     expr: up{job="gpu-exporter"} == 0
     for: 2m
     labels:
       severity: critical
     annotations:
       summary: "GPU node {{ $labels.instance }} is down"
       description: "GPU monitoring unavailable for {{ $labels.instance }}"
   
   - alert: HighGPUTemperature
     expr: gpu_temperature_celsius > 85
     for: 5m
     labels:
       severity: warning
     annotations:
       summary: "GPU {{ $labels.gpu_id }} temperature high (> 85°C)"
       description: "GPU temperature is {{ $value }}°C"
   ```

5. **Database Issues**:
   ```yaml
   - alert: DatabaseConnectionPoolExhausted
     expr: database_connections_active / database_connections_max > 0.9
     for: 5m
     labels:
       severity: warning
     annotations:
       summary: "Database connection pool near exhaustion (> 90%)"
       description: "{{ $value | humanizePercentage }} of connections in use"
   
   - alert: SlowQueries
     expr: rate(database_slow_queries_total[5m]) > 10
     for: 5m
     labels:
       severity: warning
     annotations:
       summary: "High number of slow queries (> 10/min)"
       description: "{{ $value }} slow queries per minute"
   ```

6. **Cache Degradation**:
   ```yaml
   - alert: LowCacheHitRate
     expr: cache_hit_rate < 0.7
     for: 15m
     labels:
       severity: warning
     annotations:
       summary: "Low cache hit rate (< 70%)"
       description: "Cache hit rate is {{ $value | humanizePercentage }}"
   ```

7. **Data Drift**:
   ```yaml
   - alert: DataDriftDetected
     expr: data_drift_score > 0.5  # KL divergence threshold
     for: 1h
     labels:
       severity: warning
     annotations:
       summary: "Data drift detected (KL divergence > 0.5)"
       description: "Feature {{ $labels.feature_name }} drift score is {{ $value }}"
   ```

**Alert Routing** (Alertmanager):
```yaml
# alertmanager.yml
route:
  receiver: 'default'
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty'
      continue: true
    
    - match:
        severity: critical
      receiver: 'slack-critical'
    
    - match:
        severity: warning
      receiver: 'slack-warnings'

receivers:
  - name: 'default'
    email_configs:
      - to: 'team@example.com'
  
  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: '<pagerduty-key>'
  
  - name: 'slack-critical'
    slack_configs:
      - api_url: '<slack-webhook-url>'
        channel: '#alerts-critical'
        title: 'CRITICAL: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
  
  - name: 'slack-warnings'
    slack_configs:
      - api_url: '<slack-webhook-url>'
        channel: '#alerts-warnings'
        title: 'WARNING: {{ .GroupLabels.alertname }}'
```

**On-Call Rotation**:
- **Primary On-Call**: ML Engineer (responds to critical alerts within 15 minutes)
- **Secondary On-Call**: DevOps Engineer (escalation after 30 minutes)
- **Business Hours**: 8 AM - 6 PM weekdays (faster response expected)
- **After Hours**: Automated escalation to secondary if primary doesn't acknowledge

### 18.4 Dashboards

**Grafana Dashboards**:

1. **API Performance Dashboard**:
   - **Panels**:
     - Request rate: Graph showing requests/sec over time (per endpoint)
     - Latency heatmap: p50, p95, p99 latency (color-coded)
     - Error rate: % errors over time (threshold line at 5%)
     - Active requests: Gauge showing current concurrent requests
     - Top endpoints: Table sorted by request volume
   - **Time Range**: Last 1 hour (default), adjustable to 24h, 7d, 30d
   - **Refresh**: Auto-refresh every 30 seconds

2. **Model Performance Dashboard**:
   - **Panels**:
     - Prediction accuracy: Line graph (daily) with 92% target line
     - MAE/RMSE: Dual-axis graph showing error metrics over time
     - Confidence distribution: Histogram of confidence scores
     - Defect type distribution: Pie chart showing % per defect class
     - Predictions per hour: Bar chart showing hourly volume
     - Data drift indicators: Heatmap showing drift score per feature
   - **Alerts**: Visual indicators when accuracy <87% or drift >0.5

3. **Infrastructure Dashboard**:
   - **Panels**:
     - GPU utilization: Time series per GPU (target: 60-80%)
     - GPU memory: Stacked area chart (used/available)
     - GPU temperature: Line graph (alert threshold at 85°C)
     - CPU/Memory: Node-level resource usage
     - Pod status: Table showing pod health, restarts, age
     - Network I/O: Ingress/egress bandwidth
   - **Filters**: Select by node, namespace, pod

4. **Database Dashboard**:
   - **Panels**:
     - Connection pool: Active/idle/waiting connections over time
     - Query latency: p50, p95, p99 for SELECT, INSERT, UPDATE
     - Slow queries: Table listing queries >100ms with query text
     - Cache hit ratio: % (PostgreSQL buffer cache)
     - Replication lag: Seconds lag for read replicas
     - Disk usage: % used per tablespace
   - **Alerts**: Connection pool >90%, slow queries >10/min

5. **Business Metrics Dashboard**:
   - **Panels**:
     - Daily predictions: Count over last 30 days
     - Yield prediction vs. actual: Scatter plot (correlation)
     - Cost savings: Calculated from test time reduction (daily/monthly)
     - User adoption: Active users per week
     - Top users: Table showing prediction volume per user
     - Defect trends: Stacked bar chart showing defect types over time
   - **Audience**: Management, product owners, yield engineers

6. **Real-Time Monitoring Dashboard** (Wall Display):
   - **Panels**:
     - Large gauge: Current predictions/hour
     - Traffic light: System health (green/yellow/red based on alerts)
     - Recent predictions: Live feed of last 10 predictions
     - Alert feed: Scrolling list of active alerts
     - SLO compliance: % of requests meeting latency SLO
   - **Auto-Cycle**: Rotate through key metrics every 30 seconds

**Dashboard Access Control**:
- **Viewers**: All users (read-only access)
- **Editors**: ML Engineers, DevOps (can modify dashboards)
- **Admins**: System Admins (full access including user management)

**Exported Dashboards** (JSON):
```json
{
  "dashboard": {
    "title": "Model Performance",
    "panels": [
      {
        "title": "Prediction Accuracy",
        "targets": [
          {
            "expr": "prediction_accuracy{model_version='resnet18-v1.2'}",
            "legendFormat": "{{ product_id }}"
          }
        ],
        "thresholds": [
          {"value": 0.87, "color": "red"},
          {"value": 0.92, "color": "green"}
        ]
      }
    ]
  }
}
```

---

## 19. Risk Assessment

### 19.1 Technical Risks

**Risk 1: Model Accuracy Degradation Over Time**
- **Description**: Model trained on 2024 data may degrade as manufacturing processes change in 2025-2026
- **Likelihood**: High (process drift common in semiconductor manufacturing)
- **Impact**: High (inaccurate predictions → poor business decisions, lost cost savings)
- **Mitigation**:
  - Implement data drift detection (KL divergence monitoring)
  - Automated monthly retraining with latest data
  - A/B testing new models vs. production before promotion
  - Human-in-the-loop for low-confidence predictions
  - Quarterly model performance review with yield engineers
- **Contingency**: Rollback to previous model version, manual wafer map review as fallback

**Risk 2: Transfer Learning Not Effective for All Products**
- **Description**: Transfer from TC42x may not work well for dissimilar products (e.g., different die sizes, package types)
- **Likelihood**: Medium (some products may have unique failure modes)
- **Impact**: Medium (product-specific models require more training data, slower NPI)
- **Mitigation**:
  - Evaluate transfer learning on all product families during Phase 1
  - Maintain product-specific models where transfer fails (<85% accuracy)
  - Collect 5,000+ samples per product for fallback training
  - Fine-tune hyperparameters per product family
- **Contingency**: Train from scratch for products where transfer learning accuracy <80%

**Risk 3: GPU Hardware Failures**
- **Description**: GPU failures disrupt inference, especially if only 1-2 GPUs in production
- **Likelihood**: Low (but impactful if occurs)
- **Impact**: High (no predictions available, revert to manual analysis)
- **Mitigation**:
  - Deploy 4 GPU instances (redundancy)
  - Automatic failover to CPU inference (10× slower but functional)
  - GPU health monitoring (temperature, memory errors, CUDA errors)
  - Spare GPU capacity (1-2 GPUs idle for failover)
- **Contingency**: CPU inference, reduce batch size, queue predictions for GPU recovery

**Risk 4: Database Scalability Bottleneck**
- **Description**: PostgreSQL may struggle with 100M+ predictions as system scales
- **Likelihood**: Medium (if usage grows 10× beyond projections)
- **Impact**: Medium (slow queries, timeouts, degraded user experience)
- **Mitigation**:
  - Table partitioning by month (predictions table)
  - Read replicas for SELECT queries (3 replicas planned)
  - Connection pooling (PgBouncer)
  - Archive old predictions to S3 (>1 year old)
  - Consider move to time-series DB (TimescaleDB, InfluxDB) if needed
- **Contingency**: Emergency database upgrade, temporary write throttling

**Risk 5: ONNX Compatibility Issues**
- **Description**: PyTorch → ONNX export may fail for custom layers or future architectures
- **Likelihood**: Low (ResNet is well-supported)
- **Impact**: Medium (cannot deploy optimized model, slower inference)
- **Mitigation**:
  - Test ONNX export in CI/CD pipeline
  - Maintain PyTorch fallback for inference
  - Use TorchScript as alternative to ONNX
  - Limit model architectures to ONNX-compatible (ResNet, EfficientNet, ViT)
- **Contingency**: Deploy PyTorch model directly (2-3× slower but functional)

**Risk 6: Data Privacy / Security Breach**
- **Description**: Wafer data, predictions, or models leaked externally
- **Likelihood**: Low (with proper security measures)
- **Impact**: Critical (competitive disadvantage, regulatory fines, customer loss)
- **Mitigation**:
  - Encryption at rest (AES-256) and in transit (TLS 1.3)
  - RBAC with least privilege access
  - Audit logging of all data access
  - Pen testing annually, vulnerability scanning daily
  - Secrets management (HashiCorp Vault)
- **Contingency**: Incident response plan, data breach notification protocol, insurance

### 19.2 Business Risks

**Risk 1: User Adoption Lower Than Expected**
- **Description**: Engineers don't trust AI predictions, continue manual methods
- **Likelihood**: Medium (change management challenge)
- **Impact**: High (ROI not achieved, project perceived as failure)
- **Mitigation**:
  - Involve users early (pilot program, feedback loops)
  - Extensive training and onboarding
  - Show Grad-CAM explanations (build trust via interpretability)
  - Gradual rollout (assisted prediction before automation)
  - Measure and communicate success stories (time saved, yield improved)
- **Contingency**: Extended pilot phase, dedicated change management resources

**Risk 2: Business Case Not Realized (Cost Savings <$3M)**
- **Description**: Test time reduction or yield improvement lower than projected
- **Likelihood**: Medium (conservative estimates, but unknowns remain)
- **Impact**: High (project ROI questioned, future funding at risk)
- **Mitigation**:
  - Conservative Phase 5 rollout (start with high-confidence cases only)
  - Detailed tracking of cost savings (A/B testing, control groups)
  - Diversify value: not just cost savings, also engineer productivity, NPI acceleration
  - Quarterly business reviews with stakeholders
- **Contingency**: Pivot to different value drivers (e.g., quality improvement, faster debug)

**Risk 3: Competition / Vendor Solutions Emerge**
- **Description**: Equipment vendors (Advantest, Teradyne) release similar AI/ML features
- **Likelihood**: Medium (industry trend toward AI in test)
- **Impact**: Medium (internal solution less differentiated)
- **Mitigation**:
  - Build proprietary domain expertise (wafer map patterns, transfer learning for semiconductors)
  - Patent key innovations (transfer learning for yield, Grad-CAM for defect localization)
  - Faster iteration than vendors (internal team more agile)
  - Integration with internal systems (MES, yield databases)
- **Contingency**: Collaborate with vendors (provide training data, integrate their models)

**Risk 4: Regulatory / Compliance Changes**
- **Description**: New regulations on AI/ML in manufacturing (e.g., model explainability requirements)
- **Likelihood**: Low (but increasing regulatory scrutiny on AI)
- **Impact**: Medium (need to retrofit explainability, auditing)
- **Mitigation**:
  - Proactive explainability (Grad-CAM, SHAP already planned)
  - Comprehensive audit logs (model versions, predictions, training data)
  - Stay informed on industry regulations (ISO, SEMI standards)
- **Contingency**: Engage legal/compliance teams, retrofit features as needed

**Risk 5: Key Personnel Turnover**
- **Description**: ML engineers, yield engineers with domain knowledge leave
- **Likelihood**: Medium (competitive job market)
- **Impact**: High (project delays, knowledge loss)
- **Mitigation**:
  - Comprehensive documentation (PRD, code docs, runbooks)
  - Knowledge sharing sessions (weekly ML team meetings)
  - Cross-training (multiple engineers familiar with each component)
  - Competitive compensation and career development
- **Contingency**: Contractor support, extended timelines, hire replacements quickly

### 19.3 Mitigation Strategies

**Proactive Strategies**:

1. **Continuous Model Monitoring**:
   - Daily accuracy tracking (compare predictions vs. actual yield)
   - Weekly data drift reports (distribution shifts in input features)
   - Monthly model retraining (keep model fresh with latest data)
   - Quarterly model architecture review (consider newer architectures: ViT, ConvNeXt)

2. **Robust Testing & Validation**:
   - 90%+ code coverage for core modules
   - Integration tests for all critical workflows
   - Load testing before each production release
   - Canary deployments (10% traffic for 24 hours before full rollout)
   - Automated rollback on error rate >5%

3. **User Feedback Loops**:
   - In-app feedback buttons (thumbs up/down on predictions)
   - Quarterly user satisfaction surveys (NPS score)
   - Monthly meetings with power users (yield engineers)
   - Slack channel for feature requests and bug reports
   - Dedicated product manager for user advocacy

4. **Incremental Rollout**:
   - Phase 1-2: Offline validation (no user impact)
   - Phase 3: Shadow mode (validate accuracy, no user actions)
   - Phase 4: Limited beta (10 users, feedback)
   - Phase 5: Assisted prediction (50% users, human oversight)
   - Phase 6: Full automation (100% users, monitored closely)

5. **Fallback Mechanisms**:
   - GPU failure → CPU inference
   - Model unavailable → rule-based estimate (PASS/FAIL ratio)
   - Database down → cached predictions, queue writes
   - High error rate → automatic rollback to previous version

**Reactive Strategies**:

1. **Incident Response Plan**:
   - On-call rotation (24/7 coverage for critical alerts)
   - Runbooks for common incidents (GPU failure, database timeout, model accuracy drop)
   - Post-mortem process (within 24 hours of incident, action items tracked)
   - Blameless culture (focus on systemic fixes, not individual fault)

2. **Data Backup & Recovery**:
   - Daily database backups (full backup + incremental)
   - Wafer map images replicated to 3 regions
   - Model checkpoints versioned with DVC (easy rollback)
   - RPO: <4 hours (max data loss)
   - RTO: <2 hours (max downtime)

3. **Security Incident Response**:
   - Incident response team (security, legal, PR, engineering)
   - 72-hour breach notification (GDPR compliance)
   - Forensics tools (preserve logs, analyze attack vectors)
   - Customer communication plan (transparency, remediation steps)

4. **Business Continuity**:
   - Document all critical processes (no single point of failure)
   - Cross-train team members (knowledge redundancy)
   - Contractor bench (rapid scale-up if needed)
   - Budget reserve (15% contingency for unforeseen costs)

---

## 20. Timeline & Milestones

### 20.1 Phase Breakdown

**Phase 0: Project Kickoff** (Weeks 1-2, Dec 2025)
- Assemble team: 4-5 FTE (ML Engineer, Computer Vision Engineer, Backend Engineer, DevOps, Product Manager)
- Stakeholder alignment: Present PRD, get approval on scope and budget
- Infrastructure setup: Provision GPU servers (4× A10), PostgreSQL, MinIO, Kubernetes cluster
- Data collection: Identify historical STDF files (50K wafers), secure access permissions
- Success Criteria: Team onboarded, infrastructure ready, data accessible

**Phase 1: Data Preparation & Model Prototyping** (Weeks 3-6, Jan 2026)
- STDF parsing: Develop STDF parser (pystdf), extract die coordinates and bins
- Wafer map generation: Build wafer map renderer (300×300 RGB images)
- Dataset creation: Generate 70K wafer maps (train/val/test = 70/15/15)
- Manual labeling: Yield engineers label 10K wafers with defect types (8 classes)
- ResNet baseline: Train ResNet-18 from scratch (no transfer learning) on 50K samples
- Transfer learning prototype: Fine-tune ResNet-18 with ImageNet weights, compare to baseline
- Success Criteria: 70K wafer map dataset, baseline model >85% accuracy, transfer learning >88% accuracy

**Phase 2: Model Development & Optimization** (Weeks 7-12, Feb-Mar 2026)
- Progressive fine-tuning: Implement 3-phase training (freeze → unfreeze last block → full)
- Domain adaptation: Batch normalization adaptation, discriminative learning rates
- Data augmentation: Implement rotations, flips, color jitter, elastic transforms
- Hyperparameter tuning: Optuna for LR, batch size, weight decay, epochs
- Grad-CAM implementation: Visualize attention maps for explainability
- Multi-resolution support: Device-level (300×300), die-level (32×32), full-wafer (1024×1024)
- ONNX export: PyTorch → ONNX → TensorRT optimization (FP16)
- Success Criteria: ResNet-18 >92% accuracy, ResNet-50 >94% accuracy, ONNX inference <200ms p95

**Phase 3: API & Infrastructure Development** (Weeks 13-18, Apr-May 2026)
- FastAPI backend: REST endpoints (/predict, /grad-cam, /similarity-search, /retrain)
- Database schema: PostgreSQL tables (wafers, predictions, models, experiments)
- MinIO integration: Object storage for wafer maps, model checkpoints, Grad-CAM heatmaps
- Redis caching: Prediction cache (TTL=1 hour), model metadata cache
- MLflow setup: Experiment tracking, model registry, versioning
- GPU inference service: ONNX Runtime + TensorRT, dynamic batching
- Kubernetes deployment: Helm charts, autoscaling, load balancing
- Success Criteria: API functional, <500ms p95 latency, 99% uptime, GPU inference working

**Phase 4: Frontend & User Experience** (Weeks 19-24, May-Jun 2026)
- React frontend: Wafer map viewer (zoom/pan), prediction dashboard, Grad-CAM overlay
- User authentication: OAuth2 + JWT, RBAC (model_user, yield_engineer, ml_engineer, admin)
- Training monitoring: Embedded TensorBoard, real-time progress tracking
- Similarity search UI: Upload wafer map → find top-10 similar historical wafers
- Admin panel: Model management, dataset versioning, user management
- Mobile-responsive: Tablet and mobile view (read-only)
- Success Criteria: UI functional, <2 sec page load, >4.0/5.0 usability rating from pilot users

**Phase 5: Testing & Validation** (Weeks 25-30, Jul-Aug 2026)
- Unit testing: 90%+ code coverage, pytest for backend, Jest for frontend
- Integration testing: E2E workflows (STDF upload → prediction → DB storage)
- Performance testing: Load tests (100, 200, 500 users), stress tests, soak tests (24 hours)
- Security testing: SAST (SonarQube), DAST (OWASP ZAP), dependency scanning (Snyk)
- User acceptance testing: 10 pilot users (yield engineers) test for 2 weeks, provide feedback
- Model validation: Holdout test set (10K wafers), temporal validation (2024 data → 2025 validation)
- Success Criteria: Zero critical bugs, >92% model accuracy on test set, <200ms p95 latency, >4.2/5.0 user satisfaction

**Phase 6: Pilot Deployment** (Weeks 31-38, Sep-Oct 2026)
- Staging deployment: Full system deployed to staging environment
- Offline validation: Batch predictions on 10K historical wafers, compare to actual yield
- Shadow mode: Run predictions in production (not visible to users), log for analysis
- Limited beta: 10 pilot users, predictions marked "Beta - For Review Only"
- Feedback iteration: Weekly meetings with pilot users, fix bugs, improve UI/UX
- A/B testing: Compare transfer learning vs. from-scratch models
- Success Criteria: Accuracy >90% on live data, zero production incidents, 80% pilot user adoption

**Phase 7: Production Rollout** (Weeks 39-48, Nov 2026-Jan 2027)
- Gradual rollout: 10% → 25% → 50% → 75% → 100% of users over 10 weeks
- Assisted prediction: Predictions provided as recommendations, engineer makes final decision
- Adaptive test termination: Enable for high-confidence cases (confidence >95%, yield >85% or <20%)
- Monitor impact: Track test time reduction, DPPM, cost savings, user satisfaction
- Continuous improvement: Monthly model retraining, quarterly feature releases
- Documentation: User guides, video tutorials, API documentation, runbooks
- Success Criteria: 100% rollout, 20% test time reduction, DPPM increase <1%, $2M+ cost savings (year 1)

**Phase 8: Scale & Optimization** (Months 13-18, Feb-Jul 2027)
- Multi-product expansion: Deploy to 15+ product families (TC41x, TC42x, TC43x, etc.)
- Multi-modal fusion: Combine wafer maps + parametric trends + shmoo plots
- Advanced features: Active learning, temporal trend analysis, counterfactual explanations
- Performance optimization: Prune models (30% sparsity), distillation (ResNet-50 → ResNet-18)
- Multi-region deployment: US, EU, Asia-Pacific (latency <100ms globally)
- Success Criteria: 15+ products onboarded, >95% multi-modal accuracy, global deployment, $5M+ annual cost savings

### 20.2 Key Milestones

| Milestone | Target Date | Deliverable | Success Criteria |
|-----------|-------------|-------------|------------------|
| **M1: Project Kickoff** | Week 2 (Dec 2025) | Team assembled, infrastructure ready | 4-5 FTE onboarded, GPU servers provisioned |
| **M2: Dataset Ready** | Week 6 (Jan 2026) | 70K wafer maps labeled | Train/val/test splits (70/15/15), 10K labeled defect types |
| **M3: Baseline Model** | Week 8 (Feb 2026) | ResNet-18 trained from scratch | >85% accuracy on validation set |
| **M4: Transfer Learning Model** | Week 12 (Mar 2026) | ResNet-18 fine-tuned (3 phases) | >92% accuracy, <200ms inference |
| **M5: API Functional** | Week 18 (May 2026) | FastAPI backend deployed | /predict, /grad-cam endpoints working, <500ms p95 |
| **M6: Frontend Complete** | Week 24 (Jun 2026) | React UI deployed | Wafer map viewer, prediction dashboard, Grad-CAM overlay |
| **M7: Testing Complete** | Week 30 (Aug 2026) | All tests passing | 90%+ coverage, zero critical bugs, >4.2/5.0 user rating |
| **M8: Pilot Deployment** | Week 34 (Sep 2026) | 10 pilot users onboarded | 80% adoption, >90% accuracy on live data |
| **M9: Shadow Mode** | Week 36 (Oct 2026) | Production predictions (not visible) | Accuracy >90%, data drift <0.3 KL divergence |
| **M10: 50% Rollout** | Week 42 (Dec 2026) | 50% users using assisted prediction | 15% test time reduction, zero safety incidents |
| **M11: 100% Rollout** | Week 48 (Jan 2027) | Full production deployment | 20% test time reduction, $2M+ cost savings |
| **M12: Multi-Product Expansion** | Month 18 (Jul 2027) | 15+ products onboarded | All products >90% accuracy, $5M+ annual savings |

**Critical Path**:
1. Data preparation (Weeks 3-6) → Blocks model development
2. Model development (Weeks 7-12) → Blocks API development
3. API development (Weeks 13-18) → Blocks frontend development
4. Testing (Weeks 25-30) → Blocks pilot deployment
5. Pilot deployment (Weeks 31-38) → Blocks production rollout

**Dependencies**:
- GPU procurement: Must complete Week 1 (lead time: 2-4 weeks)
- Data access: Secure permissions Week 1 (legal approval may delay)
- Yield engineer labeling: 10K labels needed Week 5 (20 hours/engineer, 5 engineers = 1 week)
- Security review: Required before production deployment (Week 30)
- Change management approval: Required before adaptive test termination (Week 42)

**Risk Buffer**:
- 15% time contingency built into each phase
- Phase 1-3: Technical risks (data quality, model accuracy) → 2-week buffer
- Phase 4-5: Integration risks (API/UI bugs) → 2-week buffer
- Phase 6-8: Business risks (user adoption, change management) → 4-week buffer

---

## 21. Success Metrics & KPIs

### 21.1 Measurable Targets

**ML Model Performance**:
- ✅ **Yield Prediction Accuracy**: >92% on holdout test set (5,000 wafers)
  - Baseline: N/A (new capability)
  - Year 1 Target: 92%
  - Year 2 Target: 95% (with multi-modal fusion)
  - Measurement: Monthly evaluation on rolling 30-day window
  
- ✅ **Early Prediction Accuracy** (5-10% test completion): >88%
  - Baseline: N/A
  - Year 1 Target: 88%
  - Year 2 Target: 90%
  - Measurement: Compare early predictions vs. final yield
  
- ✅ **Defect Classification Accuracy**: >90% macro F1-score
  - Baseline: Manual classification (70% inter-rater agreement)
  - Year 1 Target: 90%
  - Year 2 Target: 93%
  - Measurement: Weekly evaluation against yield engineer labels
  
- ✅ **Transfer Learning Efficiency**: >85% accuracy with 500 samples (vs. 10,000 from scratch)
  - Baseline: From-scratch requires 10,000 samples for 92% accuracy
  - Year 1 Target: 500 samples → 85% accuracy
  - Year 2 Target: 300 samples → 88% accuracy
  - Measurement: Evaluate on NPI products during ramp

**Business Impact**:
- ✅ **Test Time Reduction**: 30-40% average across product portfolio
  - Baseline: 10-15 minutes average test time per device
  - Year 1 Target: 6-10 minutes (30-40% reduction)
  - Year 2 Target: 5-8 minutes (40-50% reduction)
  - Measurement: Monthly test time tracking per product
  
- ✅ **Cost Savings**: $3-5M annual test cost reduction
  - Baseline: $12M annual test spend
  - Year 1 Target: $3M savings (25% reduction)
  - Year 2 Target: $5M savings (40% reduction)
  - Measurement: Quarterly cost analysis (test time × hourly tester rate)
  
- ✅ **Yield Improvement**: +2-5 percentage points through optimized limits
  - Baseline: 85% final test yield (TC42x)
  - Year 1 Target: +2 pp (87% yield)
  - Year 2 Target: +5 pp (90% yield)
  - Measurement: Lot-by-lot yield tracking, control group comparison
  
- ✅ **NPI Acceleration**: 6-month reduction in new product yield ramp
  - Baseline: 12 months PR7 → Production
  - Year 1 Target: 9 months (3-month reduction)
  - Year 2 Target: 6 months (6-month reduction)
  - Measurement: Track NPI milestone dates
  
- ✅ **DPPM Reduction**: -30% field returns with spatial failure signatures
  - Baseline: 50 PPM with spatial signatures (edge defects, center clusters)
  - Year 1 Target: 40 PPM (-20%)
  - Year 2 Target: 35 PPM (-30%)
  - Measurement: Quarterly field failure analysis

**System Performance**:
- ✅ **Inference Latency**: <200ms p95 on GPU
  - Baseline: N/A (new system)
  - Target: <200ms p95 (single prediction)
  - Stretch: <150ms p95
  - Measurement: Prometheus metrics, daily dashboard
  
- ✅ **System Uptime**: >99.5% monthly availability
  - Baseline: N/A
  - Target: 99.5% (max 3.6 hours downtime/month)
  - Stretch: 99.9% (max 43 minutes downtime/month)
  - Measurement: Uptime monitoring (Prometheus, PagerDuty)
  
- ✅ **Throughput**: 1,000 wafers/hour batch processing
  - Baseline: Manual review = 10 wafers/hour per engineer
  - Target: 1,000 wafers/hour (100× improvement)
  - Stretch: 2,000 wafers/hour
  - Measurement: Load testing, production monitoring

**User Adoption**:
- ✅ **Active Users**: 50+ engineers using platform weekly
  - Baseline: 0 (new platform)
  - Month 6 (Pilot): 10 users
  - Month 12 (Full Rollout): 50 users
  - Month 24: 100 users (expanded to other product lines)
  - Measurement: Analytics dashboard (daily active users)
  
- ✅ **Prediction Volume**: 50,000+ predictions/month
  - Baseline: 0
  - Month 6: 5,000 predictions/month
  - Month 12: 50,000 predictions/month
  - Month 24: 100,000 predictions/month
  - Measurement: Database query (count predictions per month)
  
- ✅ **User Satisfaction**: >4.2/5.0 average rating
  - Baseline: N/A
  - Pilot (Month 6): >4.0/5.0
  - Year 1 (Month 12): >4.2/5.0
  - Year 2 (Month 24): >4.5/5.0
  - Measurement: Quarterly surveys (NPS, satisfaction scores)
  
- ✅ **Adoption Rate**: 80% of yield engineers use platform for >50% of wafer map reviews
  - Baseline: 0% (manual only)
  - Month 12: 60%
  - Month 24: 80%
  - Measurement: Survey + analytics (wafer map views)

**Operational Excellence**:
- ✅ **Model Retraining Frequency**: Monthly automated retraining
  - Target: 12 retraining runs per year (monthly)
  - Measurement: MLflow experiment tracking
  
- ✅ **Data Drift Detection**: Alert within 24 hours of drift >0.5 KL divergence
  - Target: Zero missed drift events
  - Measurement: Prometheus alerts, incident tracking
  
- ✅ **Mean Time to Recovery (MTTR)**: <30 minutes for critical incidents
  - Target: <30 minutes (from alert to resolution)
  - Measurement: PagerDuty incident duration
  
- ✅ **False Positive Rate**: <5% of predictions flagged as incorrect by users
  - Target: <5%
  - Stretch: <2%
  - Measurement: User feedback ("Report Incorrect Prediction" button)

**ROI Metrics**:
- ✅ **Project ROI**: 2-4× return in Year 1
  - Investment: $1.0-1.2M (personnel + infrastructure)
  - Return Year 1: $3-5M (test cost savings + yield improvement)
  - ROI: 2.5-4.2× (250-420%)
  - Measurement: Finance review (quarterly)
  
- ✅ **Payback Period**: 8-12 months
  - Target: <12 months from project start to break-even
  - Measurement: Cumulative savings vs. cumulative cost
  
- ✅ **Cost per Prediction**: <$0.10
  - Calculation: Monthly infrastructure cost / Monthly predictions
  - Target: <$0.10 per prediction
  - Measurement: Monthly cost analysis

---

## 22. Appendices & Glossary

### 22.1 Technical Background

**Transfer Learning Overview**:
Transfer learning is a machine learning technique where a model trained on one task (source domain) is repurposed for a related task (target domain). In this project:
- **Source Domain**: ImageNet (1.28M natural images, 1000 object classes)
- **Target Domain**: Semiconductor wafer maps (spatial defect patterns, 8 defect types)
- **Hypothesis**: Low-level visual features (edges, textures, blobs) learned from ImageNet transfer to wafer maps
- **Advantage**: Achieves >85% accuracy with 500 wafer samples vs. 10,000+ required for training from scratch

**ResNet Architecture**:
ResNet (Residual Network) introduced by He et al. (2015) uses skip connections to enable training very deep networks (18-152 layers):
- **Key Innovation**: Residual connections `F(x) + x` allow gradients to flow directly through network
- **ResNet-18**: 18 layers, 11.7M parameters, 1.8 GFLOPs
- **ResNet-50**: 50 layers, 25.6M parameters, 4.1 GFLOPs
- **Blocks**: BasicBlock (ResNet-18/34), Bottleneck (ResNet-50/101/152)

**Grad-CAM (Gradient-weighted Class Activation Mapping)**:
Grad-CAM generates heatmaps showing which regions of an input image are important for a prediction:
- **Method**: Compute gradient of predicted class w.r.t. final convolutional layer activations
- **Output**: Heatmap highlighting important spatial regions (e.g., edge dies for EdgeEffect prediction)
- **Use Case**: Explainability - show yield engineers why model predicts low yield

**Domain Adaptation Techniques**:
- **Batch Normalization Adaptation**: Update BN statistics (running_mean, running_var) on target domain data
- **Discriminative Learning Rates**: Lower LR for early layers (preserve ImageNet features), higher LR for late layers (adapt to wafer maps)
- **Adversarial Domain Adaptation**: Train domain discriminator to distinguish source vs. target, feature extractor trained to fool discriminator (domain-invariant features)

### 22.2 References

**Academic Papers**:
1. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR 2016.
   - Original ResNet paper, 72,000+ citations
   
2. Selvaraju, R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV 2017.
   - Grad-CAM technique for model interpretability
   
3. Yosinski, J., et al. (2014). "How transferable are features in deep neural networks?" NIPS 2014.
   - Empirical study of transfer learning effectiveness
   
4. Ganin, Y., et al. (2016). "Domain-Adversarial Training of Neural Networks." JMLR 2016.
   - Adversarial domain adaptation framework
   
5. Tan, M., & Le, Q. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." ICML 2019.
   - Alternative architecture for future exploration

**Industry Standards**:
1. JEDEC JESD22-A115: "Wafer-Level Defect Inspection"
2. SEMI E142: "Specification for Wafer Mapping"
3. IEEE 1671 (STDF): "Standard Test Data Format"
4. ISO/IEC 27001: "Information Security Management"
5. OWASP Top 10: Web application security best practices

**Technical Documentation**:
1. PyTorch Documentation: https://pytorch.org/docs/
2. torchvision ResNet: https://pytorch.org/vision/stable/models.html#resnet
3. ONNX Runtime: https://onnxruntime.ai/docs/
4. TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/
5. MLflow: https://mlflow.org/docs/latest/index.html
6. Prometheus: https://prometheus.io/docs/
7. Kubernetes: https://kubernetes.io/docs/

### 22.3 Future Enhancements with P16 ML Data Pipeline

The Enterprise ML Data Pipeline Platform (P16) can extend this project with production-scale infrastructure for real-time deployment and distributed processing:

**Real-time Ingestion (Apache Kafka)**:
- Stream STDF files from testers → Kafka topics → immediate yield predictions
- Enable live wafer map generation as tests complete (no batch delays)
- Trigger predictions at 5%, 10%, 25% test completion milestones
- Support 1000s STDFs/day ingestion with <1 min latency

**Distributed Processing (Apache Spark + Databricks)**:
- Process 10,000+ wafers/day in parallel (100× faster than single-server Pandas)
- Distributed wafer map generation using PySpark UDFs
- Feature extraction at scale (spatial statistics, parametric aggregations)
- GPU-accelerated batch inference across Spark cluster

**Feature Store (Delta Lake)**:
- Versioned feature tables: `wafer_features`, `parametric_stats`, `spatial_patterns`
- ACID transactions ensure consistent training/serving data
- Time-travel capability: reproduce predictions from 6 months ago
- Shared features across P01 (XGBoost), P02 (ResNet), P06 (LSTM) projects

**Experiment Tracking (MLflow)**:
- Centralized model registry for ResNet-18/ResNet-50 versions
- Track transfer learning experiments (freeze layers, learning rates, augmentation strategies)
- A/B testing: compare ImageNet vs. semiconductor-pretrained models
- Automated model promotion: dev → staging → production with approval gates

**Orchestration (Apache Airflow)**:
- DAG workflow: Ingest STDFs → Generate wafer maps → Extract features → Predict yield → Alert engineers
- Scheduled retraining: weekly model updates with latest wafer data
- Data quality checks: validate STDF completeness before processing
- Automated rollback if prediction accuracy drops <90%

**Model Serving (FastAPI + MLflow)**:
- Production API: `POST /api/v1/predict/yield` with <100ms latency
- Serve ONNX/TensorRT optimized models (5× faster than PyTorch)
- Auto-scaling: 2-20 replicas based on request load
- Multi-model serving: ResNet-18 (fast), ResNet-50 (accurate) based on priority

**Example Use Cases**:
- **Adaptive Test Termination**: Kafka streams partial STDF → Spark generates wafer map → ResNet predicts 88% yield at 10% completion → Auto-terminate test (save 40 minutes per lot) → FastAPI alerts tester to stop
- **Real-time Yield Dashboard**: Databricks notebook displays live predictions across all test sites, refreshed every 5 minutes
- **Automated Retraining**: Airflow triggers weekly retraining when 10,000 new wafers accumulated in Delta Lake, deploys champion model if >2% accuracy improvement
- **Cross-Product Transfer Learning**: MLflow tracks ResNet-18 trained on TC3x family → fine-tune on TC4x with 500 samples (saved in feature store) → deploy in 2 days vs. 2 weeks from scratch

**Integration Timeline**:
- **Phase 1** (Month 1-2): Kafka ingestion + Spark processing
- **Phase 2** (Month 3-4): Delta Lake feature store + MLflow tracking
- **Phase 3** (Month 5-6): Airflow orchestration + FastAPI serving
- **Phase 4** (Month 7+): Production deployment with A/B testing

### 22.4 Glossary

**ATE (Automatic Test Equipment)**: Machines that test semiconductor devices (Advantest V93000, Teradyne)

**Bin**: Classification code for die test results (Bin 1 = PASS, Bins 2-255 = various FAIL categories)

**BGA (Ball Grid Array)**: Package type with solder balls on bottom (BGA436 = 436 pins)

**DPPM (Defective Parts Per Million)**: Quality metric for field failures

**Grad-CAM**: Gradient-weighted Class Activation Mapping (explainability technique)

**ImageNet**: Large-scale image dataset (1.28M images, 1000 classes) used for pre-training

**IDDQ (Quiescent Supply Current)**: Parametric test measuring leakage current

**MCU (Microcontroller Unit)**: Single-chip computer (TC3x, TC4x families)

**NOTEST**: Die positions not tested (e.g., outside wafer boundary, probe card defects)

**NPI (New Product Introduction)**: Process of bringing new product from design to production

**ONNX (Open Neural Network Exchange)**: Cross-platform model format

**PRR (Part Result Record)**: STDF record containing per-die test results (x, y, bin)

**Pareto (bin)**: Statistical distribution of bins (e.g., 80% Bin 1, 15% Bin 5, 5% others)

**ResNet (Residual Network)**: Deep CNN architecture with skip connections

**SPC (Statistical Process Control)**: Monitoring manufacturing process stability

**STDF (Standard Test Data Format)**: IEEE 1671 format for test data

**TensorRT**: NVIDIA library for optimizing deep learning inference

**Vth (Threshold Voltage)**: Voltage at which transistor turns on (key parametric test)

**Wafer Map**: 2D visualization of die pass/fail status across wafer

**Yield**: Percentage of good dies (e.g., 87.3% = 4,365 PASS out of 5,000 total)

---

**End of Product Requirements Document**

**Version**: v1.0  
**Last Updated**: 2025-12-04  
**Total Sections**: 22  
**Total Lines**: 3,700+  
**Status**: Complete for Review

