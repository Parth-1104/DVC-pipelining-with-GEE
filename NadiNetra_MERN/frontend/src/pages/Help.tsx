import React from 'react';
import { HelpCircle, Book, MessageCircle, Phone, Mail, ExternalLink } from 'lucide-react';

export default function Help() {
  const faqs = [
    {
      question: "What is the Water Quality Index (WQI)?",
      answer: "The Water Quality Index is a comprehensive measure that combines various parameters including pH levels, dissolved oxygen, turbidity, pollutants, and biodiversity to provide a single number representing the overall water quality. A higher WQI indicates better water quality."
    },
    {
      question: "How often is the data updated?",
      answer: "Our monitoring systems update data in real-time, with measurements taken every 15 minutes for critical parameters. Historical data and analytics are processed and updated daily."
    },
    {
      question: "What do the different alert levels mean?",
      answer: "We use a three-tier alert system: Green (Normal), Yellow (Attention Required), and Red (Critical). These are based on WQI thresholds and sudden changes in water quality parameters."
    },
    {
      question: "How can I report water quality issues?",
      answer: "You can submit reports through the Community Reports section in the Analytics dashboard. Include specific details about the location, issue type, and any supporting information."
    }
  ];

  const guides = [
    {
      title: "Understanding Water Quality Parameters",
      description: "Learn about different parameters we measure and their significance.",
      icon: Book
    },
    {
      title: "Using the Analytics Dashboard",
      description: "A comprehensive guide to interpreting graphs and data visualizations.",
      icon: MessageCircle
    },
    {
      title: "Community Reporting Guidelines",
      description: "Best practices for submitting accurate and helpful reports.",
      icon: HelpCircle
    }
  ];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Help Center</h1>
        <p className="mt-2 text-gray-600">Find answers to common questions and learn how to use Nadinetra effectively</p>
      </div>

      {/* Quick Links */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <a href="#contact" className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
          <Phone className="h-6 w-6 text-blue-600 mb-3" />
          <h3 className="font-semibold text-gray-900">Contact Support</h3>
          <p className="text-sm text-gray-600 mt-1">Get help from our team</p>
        </a>
        <a href="#guides" className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
          <Book className="h-6 w-6 text-green-600 mb-3" />
          <h3 className="font-semibold text-gray-900">User Guides</h3>
          <p className="text-sm text-gray-600 mt-1">Detailed documentation</p>
        </a>
        <a href="#faq" className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
          <HelpCircle className="h-6 w-6 text-purple-600 mb-3" />
          <h3 className="font-semibold text-gray-900">FAQs</h3>
          <p className="text-sm text-gray-600 mt-1">Common questions answered</p>
        </a>
      </div>

      {/* FAQs */}
      <div id="faq" className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">Frequently Asked Questions</h2>
        <div className="space-y-6">
          {faqs.map((faq, index) => (
            <div key={index} className="border-b border-gray-200 last:border-0 pb-6 last:pb-0">
              <h3 className="font-medium text-gray-900 mb-2">{faq.question}</h3>
              <p className="text-gray-600">{faq.answer}</p>
            </div>
          ))}
        </div>
      </div>

      {/* User Guides */}
      <div id="guides" className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">User Guides</h2>
        <div className="grid gap-6">
          {guides.map((guide, index) => (
            <div key={index} className="flex items-start gap-4 p-4 rounded-lg border border-gray-200 hover:border-blue-500 transition-colors">
              <guide.icon className="h-6 w-6 text-blue-600 mt-1" />
              <div>
                <h3 className="font-medium text-gray-900">{guide.title}</h3>
                <p className="text-sm text-gray-600 mt-1">{guide.description}</p>
                <a href="#" className="inline-flex items-center text-sm text-blue-600 hover:text-blue-800 mt-2">
                  Read more
                  <ExternalLink className="h-4 w-4 ml-1" />
                </a>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Contact Support */}
      <div id="contact" className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">Contact Support</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <Mail className="h-5 w-5 text-gray-600" />
              <div>
                <p className="font-medium text-gray-900">Email Support</p>
                <p className="text-sm text-gray-600">support@nadinetra.com</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <Phone className="h-5 w-5 text-gray-600" />
              <div>
                <p className="font-medium text-gray-900">Phone Support</p>
                <p className="text-sm text-gray-600">+91 (800) 123-4567</p>
              </div>
            </div>
          </div>
          <form className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Subject</label>
              <input
                type="text"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="What can we help you with?"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Message</label>
              <textarea
                rows={4}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Describe your issue..."
              ></textarea>
            </div>
            <button className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 transition-colors">
              Send Message
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}