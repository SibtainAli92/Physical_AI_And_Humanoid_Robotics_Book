import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Head from '@docusaurus/Head';

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();

  return (
    <header className="hero hero--primary" style={{ padding: '4rem 0' }}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>

        <div className="button-group" style={{ marginTop: '1.5rem' }}>
          <Link className="button button--secondary button--lg" to="/docs/intro">
            Read the Book
          </Link>
        </div>
      </div>
    </header>
  );
}

function ModuleCard({ title, description, link }: any) {
  return (
    <div className="col col--3 margin-bottom--lg">
      <div className="card shadow--md" style={{ height: '100%' }}>
        <div className="card__body">
          <h3>{title}</h3>
          <p>{description}</p>
        </div>

        <div className="card__footer">
          <Link className="button button--primary button--block" to={link}>
            Explore
          </Link>
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  const { siteConfig } = useDocusaurusContext();

  return (
    <Layout
      title={siteConfig.title}
      description="Physical AI & Humanoid Robotics Book – A Complete Guide from Simulation to Reality"
    >
      <Head>
        <meta name="description" content={siteConfig.tagline} />
      </Head>

      <HomepageHeader />

      <main>
        <section className="container padding-vert--xl">
          <div className="row">
            <ModuleCard
              title="Module 1 – ROS 2"
              description="The Robotic Nervous System"
              link="/docs/module1-ros2-nervous/index"
            />
            <ModuleCard
              title="Module 2 – Digital Twin"
              description="Gazebo & Unity Simulation"
              link="/docs/module2-digital-twin-simulation/index"
            />
            <ModuleCard
              title="Module 3 – NVIDIA Isaac"
              description="Platform for AI Robotics"
              link="/docs/module3-ai-brain-isaac/index"
            />
            <ModuleCard
              title="Module 4 – VLA"
              description="Vision-Language-Action"
              link="/docs/module4-vla-robotics/index"
            />
          </div>
        </section>
      </main>
    </Layout>
  );
}
