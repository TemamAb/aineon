// PLATINUM SOURCES: React Admin, Material-UI
// CONTINUAL LEARNING: User behavior learning, layout optimization

import React, { useState, useEffect, useCallback } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { SnackbarProvider } from 'notistack';
import { AuthProvider, useAuth } from './hooks/useAuth';
import { WebSocketProvider } from './hooks/useWebSocket';
import { UserPreferencesProvider } from './hooks/useUserPreferences';

// Components
import Layout from './components/Layout/Layout';
import Login from './components/Auth/Login';
import LiveMonitoring from './components/LiveMonitoring/LiveMonitoring';
import TradingParameters from './components/Trading/TradingParameters';
import AITerminal from './components/AI/AITerminal';
import WalletSecurity from './components/Security/WalletSecurity';
import ProfitWithdrawal from './components/Finance/ProfitWithdrawal';
import LoadingScreen from './components/UI/LoadingScreen';

// Services
import { dashboardService } from './services/dashboardService';
import { userBehaviorService } from './services/userBehaviorService';

// Design Tokens
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#2563eb', // Aineon blue
    },
    success: {
      main: '#10b981', // Profit green
    },
    warning: {
      main: '#f59e0b', // Warning amber
    },
    error: {
      main: '#ef4444', // Error red
    },
    background: {
      default: '#0f172a',
      paper: '#1e293b',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    },
  },
  spacing: 8,
  breakpoints: {
    values: {
      xs: 0,
      sm: 768,
      md: 1024,
      lg: 1280,
      xl: 1440,
    },
  },
});

const AppContent: React.FC = () => {
  const { user, loading, hasPermission } = useAuth();
  const [initialized, setInitialized] = useState(false);
  const [userPreferences, setUserPreferences] = useState(null);

  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Initialize real-time connections
        await dashboardService.initialize();
        
        // Load user preferences
        const preferences = await userBehaviorService.loadUserPreferences();
        setUserPreferences(preferences);
        
        // Track user behavior for continual learning
        userBehaviorService.trackAppLoad();
        
        setInitialized(true);
      } catch (error) {
        console.error('Failed to initialize app:', error);
      }
    };

    initializeApp();
  }, []);

  const handleLayoutChange = useCallback((layoutConfig: any) => {
    // Learn from user layout preferences
    userBehaviorService.recordLayoutPreference(layoutConfig);
    setUserPreferences(prev => ({ ...prev, layout: layoutConfig }));
  }, []);

  if (loading || !initialized) {
    return <LoadingScreen />;
  }

  if (!user) {
    return <Login />;
  }

  return (
    <Router>
      <Layout onLayoutChange={handleLayoutChange} userPreferences={userPreferences}>
        <Routes>
          <Route path="/" element={<Navigate to="/monitoring" replace />} />
          <Route path="/monitoring" element={<LiveMonitoring />} />
          <Route path="/trading" element={<TradingParameters />} />
          <Route path="/ai-terminal" element={<AITerminal />} />
          {hasPermission('security_access') && (
            <Route path="/security" element={<WalletSecurity />} />
          )}
          {hasPermission('finance_access') && (
            <Route path="/finance" element={<ProfitWithdrawal />} />
          )}
        </Routes>
      </Layout>
    </Router>
  );
};

const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <SnackbarProvider 
        maxSnack={3}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'right',
        }}
      >
        <AuthProvider>
          <WebSocketProvider>
            <UserPreferencesProvider>
              <AppContent />
            </UserPreferencesProvider>
          </WebSocketProvider>
        </AuthProvider>
      </SnackbarProvider>
    </ThemeProvider>
  );
};

// Performance monitoring
export default React.memo(App);
