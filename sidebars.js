/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
     {
      type: 'category',
      label: 'Physical AI & Humanoid Robotics Textbook',
      items: [
        'intro',
        {
          type: 'category',
          label: 'Module 1: ROS 2 Nervous System',
          items: [
            'module1-ros2-nervous-system/index',
            'module1-ros2-nervous-system/chapter-1-introduction',
            'module1-ros2-nervous-system/chapter-2-core-concepts',
            'module1-ros2-nervous-system/chapter-3-tools-implementation'
          ],
        },
        {
          type: 'category',
          label: 'Module 2: Digital Twin Simulation',
          items: [
            'module2-digital-twin-simulation/index',
            'module2-digital-twin-simulation/chapter-1-introduction',
            'module2-digital-twin-simulation/chapter-2-core-concepts',
            'module2-digital-twin-simulation/chapter-3-tools-implementation'
          ],
        },
        {
          type: 'category',
          label: 'Module 3: AI Brain (NVIDIA Isaac)',
          items: [
            'module3-ai-brain-isaac/index',
            'module3-ai-brain-isaac/chapter-1-introduction',
            'module3-ai-brain-isaac/chapter-2-core-concepts',
            'module3-ai-brain-isaac/chapter-3-tools-implementation'
          ],
        },
        {
          type: 'category',
          label: 'Module 4: Vision-Language-Action Robotics',
          items: [
            'module4-vla-robotics/index',
            'module4-vla-robotics/chapter-1-introduction',
            'module4-vla-robotics/chapter-2-core-concepts',
            'module4-vla-robotics/chapter-3-tools-implementation'
          ],
        },
        'weekly-roadmap',
        {
          type: 'category',
          label: 'Additional Materials',
          items: [
            'index',
            'hardware',
            'cloud',
            'final_materials',
          ],
        },
      ],
    },
  ],
};

module.exports = sidebars;