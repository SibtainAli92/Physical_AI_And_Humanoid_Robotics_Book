# 001-physical-ai-book - Specification

## 1. Objective
Normalize, clean, and update the *Physical AI & Humanoid Robotics Textbook* so the project has a single unified structure. All updates follow these rules:

- Do **not** create new folders unless allowed.
- Update existing files.
- Delete outdated folders/files.
- Rename folders to standardized module naming.
- Update sidebar, content, and theme.
- Ensure correctness when collaborator directory differs.

## 2. Directory Normalization Rules

### 2.1 Canonical Module Folders
Rename existing folders to these EXACT names:

- `docs/ros2-nervous-system` → **docs/module1-ros2-nervous-system**
- `docs/digital-twin-simulation` → **docs/module2-digital-twin-simulation**
- `docs/ai-brain-isaac` → **docs/module3-ai-brain-isaac**
- `docs/vla-robotics` → **docs/module4-vla-robotics**

No duplicate module folders should remain.

### 2.2 Deletions (Only if they exist)
Delete these files:

- `docs/final-exam.md`
- `docs/glossary.md`
- `docs/hardware.md`
- `docs/overview.md`

Delete these folders (entire folder):

- `docs/module1-ros2-nervous-system` (old)
- `docs/module2-digital-twin-simulation` (old)
- `docs/module3-ai-brain-nvidia-isaac` (old)
- `docs/module4-vision-language-action-robotics` (old)
- `specs/001-book-content-update`

## 3. Content Update Rules

### 3.1 Update intro.md
- Add title, description, book purpose.
- Add module overview preview.
- Add hero text + image placeholder.
- Add CTA: **Start Reading**.

### 3.2 Update weekly-roadmap.md
Include 1–13 week learning roadmap:
- Weeks 1–3 → ROS 2
- Weeks 4–5 → Digital Twin
- Weeks 6–7 → Isaac AI Brain
- Weeks 8–9 → VLA
- Weeks 10–13 → Capstone

### 3.3 Update "Additional Materials"
Update these files with complete content:
- `index.md`
- `cloud.md`
- `hardware.md`
- `final_materials.md`

### 3.4 Update Module Index Pages
For each:

- `module1-ros2-nervous-system/index.md`
- `module2-digital-twin-simulation/index.md`
- `module3-ai-brain-isaac/index.md`
- `module4-vla-robotics/index.md`

Add:
- Module intro
- Learning outcomes
- Tools required
- Architecture diagrams
- Chapter navigation links
- Summary

### 3.5 Review and improve module chapters
- Fix formatting
- Improve clarity
- Add missing explanations
- Add diagrams

## 4. Sidebar Update Rules

Sidebar must follow **exact structure**:

1. Physical AI & Humanoid Robotics Textbook
2. Module 1: ROS 2 Nervous System
3. Module 2: Digital Twin Simulation
4. Module 3: AI Brain (NVIDIA Isaac)
5. Module 4: Vision-Language-Action Robotics
6. Capstone Project
7. Additional Materials
8. Weekly Roadmap

Remove any leftover sidebar groups from deleted folders.

## 5. Front Page (`src/pages/index.tsx`)

Create this file (allowed):

### Must include:
- Book title
- Short description
- Start Reading → `/docs/intro`
- Four module cards
  - Icon
  - Title
  - Description
  - Link

### Frontend Requirements:
- Tailwind CSS
- Clean, modern robotics theme
- Responsive
- Neon blue/purple theme

## 6. Theme & Branding Rules
- Apply consistent theme across markdown pages.
- Use blue/purple neon accents.
- Add diagrams for key sections:
  - ROS graph
  - Digital Twin flow
  - Isaac pipeline
  - VLA model

## 7. Specs Folder Update Rules

Update the following **existing files only**:

- `specs/001-physical-ai-book/spec.md`
- `specs/001-physical-ai-book/tasks.md`
- `specs/001-physical-ai-book/plan.md`
- `specs/001-physical-ai-book/research.md`
- `specs/001-physical-ai-book/quickstart.md`

Each file must reflect:
- New folder names
- New structure
- Updated tasks
- Updated content plan
- Removal of deleted modules

## Acceptance Criteria

- [ ] Directory structure normalized with standardized module names
- [ ] Specified files deleted
- [ ] Content updated per specification
- [ ] Sidebar configuration updated
- [ ] All spec files updated with new structure
- [ ] Front page created with required elements
- [ ] QA completed with no broken links
- [ ] Docusaurus builds successfully