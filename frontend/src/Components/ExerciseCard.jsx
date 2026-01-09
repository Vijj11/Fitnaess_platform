import React from "react";
import { Link } from "react-router-dom";
import { Button, Stack, Typography } from "@mui/material";

const ExerciseCard = ({ exercise }) => (
  <>
    <Stack direction="column" className="exercise-card">
      <Link to={`/exercise/${exercise.id}`}>
        <img
          src={exercise.gifUrl}
          alt={exercise.name}
          loading="lazy"
          // marginBottom="25px"
        />

        <Typography
          ml="10px"
          color="#000"
          fontWeight="bold"
          sx={{ fontSize: { lg: "24px", xs: "20px" } }}
          mt="10px"
          pb="10px"
          textTransform="capitalize"
        >
          {exercise.name}
        </Typography>
        <Stack direction="row">
          <Button
            sx={{
              ml: "10px",
              color: "#fff",
              // marginTop: "25px",
              // marginBottom: "25px",
              background: "#FFA9A9",
              fontSize: "14px",
              borderRadius: "20px",
              textTransform: "capitalize",
              // padding: "10px",
            }}
          >
            {exercise.bodyPart}
          </Button>
          <Button
            sx={{
              ml: "21px",
              // marginBottom: "25px",
              color: "#fff",
              background: "#FCC757",
              fontSize: "14px",
              borderRadius: "20px",
              textTransform: "capitalize",
              padding: "10px",
            }}
          >
            {exercise.target}
          </Button>
        </Stack>
      </Link>
    </Stack>
  </>
);

export default ExerciseCard;
