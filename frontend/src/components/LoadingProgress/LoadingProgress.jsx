// src/components/LoadingProgress/LoadingProgress.jsx
import { Space, Typography } from "antd";
import { LoadingOutlined } from "@ant-design/icons";
import "./LoadingProgress.css";
import { useEffect, useState } from "react";

const { Text } = Typography;

const loadingStates = [
  { text: "Sent user input...", duration: 5000 },
  { text: "Validating input...", duration: 8000 },
  { text: "Processing request...", duration: 12000 },
  { text: "Retrieving data...", duration: 10000 },
  { text: "Preparing response...", duration: 15000 },
];

const LoadingProgress = ({ isLoading }) => {
  const [currentState, setCurrentState] = useState(0);

  useEffect(() => {
    if (!isLoading) {
      setCurrentState(0);
      return;
    }

    let currentTimeout;
    const moveToNextState = (stateIndex) => {
      if (stateIndex < loadingStates.length && isLoading) {
        setCurrentState(stateIndex);
        currentTimeout = setTimeout(
          () => moveToNextState(stateIndex + 1),
          loadingStates[stateIndex].duration
        );
      }
    };

    moveToNextState(0);

    return () => {
      if (currentTimeout) clearTimeout(currentTimeout);
    };
  }, [isLoading]);

  if (!isLoading) return null;

  return (
    <div className="loading-progress">
      <Space>
        <LoadingOutlined style={{ fontSize: "18px", color: "#1677ff" }} />
        <Text>{loadingStates[currentState]?.text || "Processing..."}</Text>
      </Space>
    </div>
  );
};

export default LoadingProgress;
