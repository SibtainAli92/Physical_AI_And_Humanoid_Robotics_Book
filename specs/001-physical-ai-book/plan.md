# /sp.plan — Physical AI & Humanoid Robotics Textbook

## 1. Architecture Sketch

### 1.1 Repository Structure (Final Normalized Form)


### 1.2 Architecture Goals
- Enforce a single unified folder structure.
- Ensure all collaborators normalize outdated folders.
- Improve and clean all core textbook content.
- Standardize UI/theme across all pages.
- Ensure Docusaurus builds error-free.
- Centralize updates inside `/specs/001-physical-ai-book`.

---

## 2. Section Structure

### 2.1 Core Required Sections
1. **Intro Page**
2. **Four Modules**
3. **Capstone Project**
4. **Additional Materials**
5. **Weekly Roadmap**
6. **Front Page (index.tsx)**

---

### 2.2 Detailed Section Breakdown

#### **Intro Section**
- Hero title + book purpose
- Learning path overview
- Module preview cards
- CTA button
- Diagram placeholder

#### **Module Sections (1–4)**
Each module index must include:
- Module introduction
- Learning objectives
- Tools required
- High-level architecture diagram
- Chapter list with links
- Key skills

#### **Additional Materials**
- Hardware guide
- Cloud setup
- Reference diagrams
- Quick resources

#### **Weekly Roadmap**
- 13-week detailed training structure
- Weekly tasks, tools, and diagrams
- Capstone milestones

#### **Frontend**
- Create index.tsx
- Add module cards
- Add CTA
- Ensure layout/tailwind theme matches brand

---

## 3. Research Approach

### 3.1 Internal Document Research
- Compare current docs/ folders with required canonical structure
- Identify missing files
- Detect outdated modules and deletCT required names
- All links must work
- No duplicate folders

### 4.2 Content Validation
- Intro page complete
- Weekly roadmap complete
- Additional materials updated
- Module index pages follow template
- All outdated content removed

### 4.3 File Creation Rules Validation
- Only allowed new file → `src/pages/index.tsx`
- No extra folders created
- No stray markdown files

### 4.4 Specs Folder Validation
- spec.md, plan.md, tasks.md, research.md, quickstart.md updated
- No new spec files created

---

## 5. Decisions Needing Documentation

### 5.1 Folder Naming Decisions
All collaborators must follow the four canonical module names — no exceptions.

### 5.2 File Deletion Decisions
These files and folders must be deleted because they contain outdated or duplicated content:
- final-exam.md
- glossary.md
- hardware.md
- overview.md
- old module folders
- old specs folder

### 5.3 Frontend Structure
Decision: Front page must be React + Tailwind, modern theme, module cards