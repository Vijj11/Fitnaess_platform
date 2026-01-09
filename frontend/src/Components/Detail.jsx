import React from "react";
import { Typography, Stack, Button } from "@mui/material";
// import BodyPartImage from '../assets/icons/body-part.png';
// import TargetImage from '../assets/icons/target.png';
// import EquipmentImage from '../assets/icons/equipment.png';
import BodyPartImage from "../assets1/icons/body-part.png";
import TargetImage from "../assets1/icons/target.png";
import EquipmentImage from "../assets1/icons/equipment.png";
import axios from "axios";

const Detail = ({ exerciseDetail }) => {
  const { bodyPart, gifUrl, name, target, equipment, instructions } =
    exerciseDetail;

  console.log(exerciseDetail);
  console.log(instructions);
  const extraDetail = [
    {
      icon: BodyPartImage,
      name: bodyPart,
    },
    {
      icon: TargetImage,
      name: target,
    },
    {
      icon: EquipmentImage,
      name: equipment,
    },
  ];
  const handleButtonClick = async () => {
    const modified_name = name.replace(/ /g, "_");
    try {
      const response = await axios.post(
        `http://localhost:5000/${modified_name}`
      );
      console.log(response.data); // You can handle the response data as needed
    } catch (error) {
      if (error.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        console.error("Server responded with status:", error.response.status);
        console.error("Response data:", error.response.data);
        console.error("Response headers:", error.response.headers);
      } else if (error.request) {
        // The request was made but no response was received
        console.error("No response received:", error.request);
      } else {
        // Something happened in setting up the request that triggered an Error
        console.error("Error setting up the request:", error.message);
      }
    }
  };
  return (
    <Stack
      gap="60px"
      sx={{ flexDirection: { lg: "row" }, p: "20px", alignItems: "center" }}
    >
      <img src={gifUrl} alt={name} loading="lazy" className="detail-image" />
      <Stack sx={{ gap: { lg: "35px", xs: "20px" } }}>
        <Typography
          sx={{ fontSize: { lg: "64px", xs: "30px" } }}
          fontWeight={700}
          textTransform="capitalize"
          color="black"
        >
          {name}
        </Typography>
        {/* <Typography sx={{ fontSize: { lg: "24px", xs: "18px" } }} color="white">
          Exercises keep you strong.{" "}
          <span style={{ textTransform: "capitalize", color: "black" }}>
            {name}
          </span>{" "}
          is one of the best <br /> exercises to target your {target}. It will
          help you improve your <br /> mood and gain energy.
        </Typography> */}

        <Typography
          sx={{ fontSize: { lg: "64px", xs: "30px" } }}
          fontWeight={700}
          textTransform="capitalize"
          color="black"
        >
          Instructions
        </Typography>
        {instructions &&
          instructions.map((item, index) => (
            <Typography
              key={index}
              sx={{ fontSize: { lg: "24px", xs: "18px" } }}
              color="white"
            >
              {item}
            </Typography>
          ))}

        {extraDetail?.map((item, index) => (
          <Stack key={index} direction="row" gap="24px" alignItems="center">
            <Button
              sx={{
                background: "#FFF2DB",
                borderRadius: "50%",
                width: "100px",
                height: "100px",
              }}
            >
              <img
                src={item.icon}
                alt={bodyPart}
                style={{ width: "50px", height: "50px" }}
              />
            </Button>
            <Typography
              textTransform="capitalize"
              sx={{ fontSize: { lg: "30px", xs: "20px" } }}
            >
              {item.name}
            </Typography>
          </Stack>
        ))}

        <Button
          sx={{
            background: "#FF2625",
            // borderRadius: "50%",
            width: "150px",
            height: "100px",
            color: "white",
          }}
          onClick={handleButtonClick}
        >
          Try now
        </Button>
      </Stack>
    </Stack>
  );
};

export default Detail;
