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
          type: 'link',
          label: 'Module 1: ROS 2 Nervous System',
          href: '/docs/module1-ros2-nervous-system/',
        },
        {
          type: 'link',
          label: 'Module 2: Digital Twin Simulation',
          href:  '/docs/module2-digital-twin-simulation/',  
          
        },
        {
          type: 'link',
          label: 'Module 3: AI Brain (NVIDIA Isaac)',
          href: '/docs/module3-ai-brain-isaac/'
        },
        {
          type: 'link',
          label: 'Module 4: Vision-Language-Action Robotics',
          href: '/docs/module4-vla-robotics/',
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