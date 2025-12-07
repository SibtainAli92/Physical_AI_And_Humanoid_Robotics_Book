---
id: 0002
title: Define Physical AI Robotics Book Project Spec
stage: spec
date: 2025-12-04
surface: agent
model: claude-sonnet-4-5-20250929
feature: physical-ai-robotics-book
branch: master
user: User
command: /sp.project, /sp.goal, /sp.scope, /sp.out_of_scope, /sp.modules, /sp.chapters, /sp.tasks, /sp.success_criteria, /sp.constitution
labels: ["project", "spec", "book"]
links:
  spec: specs/physical-ai-robotics-book/spec.md
  ticket: null
  adr: null
  pr: null
files:
- specs/physical-ai-robotics-book/spec.md
tests:
null
---

## Prompt

```text
Project: AI/Spec-Driven Book Creation using Docusaurus, Spec-Kit Plus, and Claude Code

/sp.goal
To write a comprehensive, clear, and technically correct book on Physical AI & Humanoid Robotics. The book will teach students how to build simulated and real-world humanoid robots using ROS 2, Gazebo, Unity, NVIDIA Isaac, and Vision-Language-Action systems.

/sp.scope
- Conceptual and practical coverage of Physical AI
- 4 core modules:
  1. ROS 2 Control Systems for humanoids
  2. Digital Twin Simulation (Gazebo & Unity)
  3. AI Perception & Navigation using NVIDIA Isaac
  4. Vision-Language-Action (VLA) systems
- Hardware and lab requirements (Workstation → Jetson → Robot)
- 13-week structured course breakdown
- Complete Capstone: Autonomous Humanoid Robot
- Docusaurus book creation and GitHub Pages deployment

/sp.out_of_scope
- Mechanical engineering design (motors, gears, PCBs)
- Full dynamics mathematics derivations
- Non-humanoid robotics such as drones or mobile bases (unless used as proxies)
- Industrial safety certification standards
- Hardware manufacturing processes

/sp.modules
Module 1 — The Robotic Nervous System (ROS 2)
- ROS 2 architecture, nodes, topics, services, actions
- rclpy for Python-driven control
- URDF for humanoid robot models
- Building and launching ROS 2 packages

Module 2 — The Digital Twin (Gazebo & Unity)
- Physics-based simulation (gravity, collisions)
- Gazebo environments for humanoids
- Unity for visualization and HRI interaction
- Sensor simulation: Lidar, Depth Cameras, IMUs

Module 3 — The AI-Robot Brain (NVIDIA Isaac)
- Isaac Sim fundamentals and photorealistic rendering
- Isaac ROS: VSLAM, mapping, navigation
- Nav2 for bipedal locomotion
- Synthetic data generation and RL training

Module 4 — Vision-Language-Action (VLA)
- Whisper for voice-to-action commands
- LLM-based planning: natural language → ROS 2 actions
- Multi-modal interaction: vision + speech + gestures
- Final Architecture: Autonomous Humanoid Workflow

/sp.chapters
1 — Introduction to Physical AI
2 — Sensors & Embodied Intelligence
3 — ROS 2 Fundamentals
4 — Building Humanoid URDF Models
5 — Gazebo Simulation Essentials
6 — Unity Visualization
7 — NVIDIA Isaac Sim for Perception
8 — Isaac ROS & Navigation
9 — Reinforcement Learning for Humanoids
10 — Vision-Language-Action Systems
11 — Whisper Voice Commands
12 — LLM-Based Planning for Robots
13 — Capstone: Autonomous Humanoid
14 — Hardware Setup & Lab Architecture
15 — Sim-to-Real Transfer Strategies

/sp.tasks
- Define book architecture and modules
- Draft high-level content for each module
- Build example ROS 2 packages and URDFs
- Prepare Gazebo/Unity simulation examples
- Integrate NVIDIA Isaac workflows
- Create VLA demo (Whisper + LLM + ROS 2)
- Write hardware setup instructions (Jetson + Sensors)
- Create Capstone blueprint
- Build Docusaurus repository
- Deploy to GitHub Pages
- Validate reproducibility of all examples

/sp.success_criteria
- Book covers all 4 modules clearly and consistently
- All ROS 2, Isaac, Gazebo steps tested on Ubuntu 22.04
- Hardware workflows verified on Jetson Orin
- Capstone robot successfully performs:
  - Perception
  - Navigation
  - Object recognition
  - Voice command → action pipeline
- Docusaurus book compiles and deploys without errors
- Students can reproduce the entire pipeline end-to-end

/sp.constitution
Core Principles:
- Accuracy from official robotics/AI documentation
- Clarity for CS/AI students
- Reproducibility with working code and simulation steps
- Rigor using established robotics frameworks

Standards:
- Cite authoritative sources (ROS docs, NVIDIA Isaac docs, research papers)
- Prefer peer-reviewed robotics and AI literature
- All code examples must run without modification
- Writing clarity: step-by-step, no ambiguity

Constraints:
- Must follow Docusaurus layout
- Must deploy on GitHub Pages
- All modules must run on Ubuntu 22.04

Success Criteria:
- End-to-end humanoid simulation built
- VLA system (Voice → Plan → Action) demonstrated
- Zero deployment errors
- Book is easy to navigate, readable, and technically sound
```

## Response snapshot

A new feature directory `specs/physical-ai-robotics-book/` was created, and the `spec.md` file was populated with the provided project details, including user stories, functional requirements, and success criteria. The PHR script failed, so the PHR was generated using an agent-native method.

## Outcome

- ✅ Impact: Project specification created, detailing the book's scope, goals, modules, chapters, and success criteria.
- 🧪 Tests: No tests were run for this action.
- 📁 Files: One file was created: `specs/physical-ai-robotics-book/spec.md`.
- 🔁 Next prompts: Proceed with creating the implementation plan (`plan.md`) and tasks list (`tasks.md`) based on this specification.
- 🧠 Reflection: Successfully parsed a complex multi-line user input and structured it into a well-formed specification document. The fallback for PHR creation worked as expected.

## Evaluation notes (flywheel)

- Failure modes observed: `create-phr.sh` script not found, requiring agent-native PHR creation.
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A
