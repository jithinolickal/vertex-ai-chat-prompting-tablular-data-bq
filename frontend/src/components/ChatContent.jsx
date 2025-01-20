// src/components/ChatContent/ChatContent.jsx
import { Avatar, Tag } from "antd";
import "./ChatContent.css";

const ChatContent = ({ messages }) => {
  return (
    <div className="chat-content">
      {messages.map((message, index) => (
        <div
          key={index}
          className={`message-container ${
            message.type === "user" ? "user-message" : "ai-message"
          }`}
        >
          {message.type === "ai" && (
            <Avatar className="ai-avatar" alt="AI">
              AI
            </Avatar>
          )}
          <div className="message-content">
            {message.visualizations && message.visualizations.length > 0 && (
              <div className="visualization-tags">
                {message.visualizations.map((viz, i) => (
                  <Tag
                    key={i}
                    color="blue"
                    style={{ cursor: "pointer", marginBottom: "8px" }}
                  >
                    {viz.type} chart
                  </Tag>
                ))}
              </div>
            )}
            {message.content}
          </div>
        </div>
      ))}
    </div>
  );
};

export default ChatContent;
