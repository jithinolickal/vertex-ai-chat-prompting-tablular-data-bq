// src/pages/ChatPage.jsx
import { useState } from "react";
import { Input, Button, Typography, TreeSelect } from "antd";
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
        title: "Azure",
        value: "recharge_azure",
        key: "recharge_azure",
      },
      {
        title: "GCP",
        value: "billing",
        key: "recharge_gcp",
      },
    ],
  },
  {
    title: "Budgets & Forecast",
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

const ChatPage = () => {
  const [hasInteracted, setHasInteracted] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState("recharge_azure"); // default value

  const handleSend = async () => {
    if (inputValue.trim() && !isLoading) {
      setIsLoading(true);

      setMessages((prev) => [
        {
          type: "user",
          content: inputValue.trim(),
        },
        ...prev,
      ]);

      try {
        const response = await axios.post(
          `http://localhost:8000/analyze?question=${encodeURIComponent(
            inputValue
          )}&data_model=${selectedModel}`
        );

        const narrative = response.data.narrative;
        const aiResponse = `${narrative.summary}\n\n${narrative.key_points
          .map((point) => `• ${point}`)
          .join("\n")}`;

        setMessages((prev) => [
          {
            type: "ai",
            content: aiResponse,
            visualizations: response.data.has_visualizations
              ? response.data.available_visualizations
              : [],
          },
          ...prev,
        ]);
      } catch (error) {
        setMessages((prev) => [
          {
            type: "ai",
            content:
              "Sorry, I encountered an error while processing your request.",
          },
          ...prev,
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
          <TreeSelect
            suffixIcon={null}
            style={{ width: 180 }}
            value={selectedModel}
            dropdownStyle={{ maxHeight: 400, overflow: "auto" }}
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
        <LoadingProgress isLoading={isLoading} />
        {hasInteracted && <ChatContent messages={messages} />}
      </div>
    </div>
  );
};
export default ChatPage;
