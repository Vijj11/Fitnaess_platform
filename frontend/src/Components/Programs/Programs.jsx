import React from "react";
import { Link } from "react-router-dom";
import "./Programs.css";
import { programsData } from "../../data/programsData";
import RightArrow from "../../assets/rightArrow.png";

const Programs = () => {
  return (
    <div className="Programs" id="programs">
      <div className="programs-header">
        <span className="stroke-text">Explore Our</span>
        <span>Programs according to </span>
        <span className="stroke-text">your fitness levels</span>
      </div>
      <div className="programs-categories">
        {programsData.map((program, index) => (
          <Link to={`/program/${program.heading.toLowerCase().replace(/\s+/g, '-')}`} key={index}>
            <div className="category">
              {program.image}
              <span>{program.heading}</span>
              <span>{program.details}</span>
              <div className="join-now">
                <span>Start Now</span>
                <img src={RightArrow} alt="Right Arrow" />
              </div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
};

export default Programs;
