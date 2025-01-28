// src/pages/ChatPage.jsx
import { useState } from "react";
import { Input, Button, Typography, TreeSelect, Select } from "antd";
import { SendOutlined, LoadingOutlined } from "@ant-design/icons";
import axios from "axios";
import ChatContent from "../components/ChatContent";
import "./ChatPage.css";
import LoadingProgress from "../components/LoadingProgress/LoadingProgress";

const { Title } = Typography;

// TreeSelect data structure
const treeData = [
  {
    title: "Recharge",
    value: "recharge",
    key: "recharge",
    children: [
      {
        title: "Cloud Consumption",
        value: "cloud_consumption",
        key: "cloud_consumption",
      },
      {
        title: "Azure",
        value: "recharge_azure",
        key: "recharge_azure",
      },
      {
        title: "GCP",
        value: "recharge_gcp",
        key: "recharge_gcp",
      },
    ],
  },
  {
    title: "Forecast",
    value: "budget",
    key: "budget",
    children: [
      {
        title: "cloud",
        value: "cloud_forecast",
        key: "cloud_forecast",
      },
    ],
  },
];

const aiModels = [
  {
    label: "Gemini",
    value: "gemini",
  },
  {
    label: "Claude",
    value: "claude",
  },
];

const ChatPage = () => {
  const [hasInteracted, setHasInteracted] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState("cloud_consumption"); // default value
  const [selectedAiModel, setSelectedAiModel] = useState("claude");

  const handleSend = async () => {
    if (inputValue.trim() && !isLoading) {
      setIsLoading(true);

      setMessages((prev) => [
        ...prev,
        {
          type: "user",
          content: inputValue.trim(),
        },
      ]);

      try {
        const response = await axios.post(
          `http://localhost:8000/analyze?question=${encodeURIComponent(
            inputValue
          )}&data_model=${selectedModel}&ai_model=${selectedAiModel}`
        );

        const narrative = response.data.narrative;
        let aiResponse;
        if (response.data.response_type === "narrative") {
          aiResponse = `${narrative.summary}\n\n${narrative.key_points
            .map((point) => `• ${point}`)
            .join("\n")}`;
        } else if (response.data.response_type === "chart") {
          aiResponse = response.data.chart_code;
        }
        
        const sql = response.data.sql;

        setMessages((prev) => [
          ...prev,
          {
            type: "ai",
            content: aiResponse,
            response_type: response.data.response_type,
            sql,
          },
        ]);
      } catch (error) {
        setMessages((prev) => [
          ...prev,
          {
            type: "ai",
            content:
              "Sorry, I encountered an error while processing your request.",
          },
        ]);
        console.error("Error:", error);
      } finally {
        setIsLoading(false);
      }

      setInputValue("");
      setHasInteracted(true);
    }
  };

  return (
    <div className="chat-page">
      <div
        className={`content-wrapper ${hasInteracted ? "has-interacted" : ""}`}
      >
        {!hasInteracted && (
          <div className="welcome-text">
            <Title level={3}>Welcome to FinOps AI Assistant</Title>
            <p>Ask me anything about your FinOps data</p>
          </div>
        )}

        {hasInteracted && <ChatContent messages={messages} />}
        <LoadingProgress isLoading={isLoading} />
        <div className="input-section">
          <Input
            placeholder="Ask anything about your FinOps data..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onPressEnter={handleSend}
            variant="borderless"
            autoFocus
            disabled={isLoading}
          />
          <Select
            style={{ width: 150 }}
            value={selectedAiModel}
            options={aiModels}
            onChange={setSelectedAiModel}
            disabled={isLoading}
          />
          <TreeSelect
            suffixIcon={null}
            style={{ width: 150 }}
            value={selectedModel}
            dropdownStyle={{ minWidth: 200, maxHeight: 400, overflow: "auto" }}
            treeData={treeData}
            placeholder="Select model"
            treeDefaultExpandAll
            onChange={setSelectedModel}
            disabled={isLoading}
          />
          <Button
            type="primary"
            icon={isLoading ? <LoadingOutlined /> : <SendOutlined />}
            onClick={handleSend}
            disabled={!inputValue.trim() || isLoading}
          />
        </div>
      </div>
    </div>
  );
};
export default ChatPage;
