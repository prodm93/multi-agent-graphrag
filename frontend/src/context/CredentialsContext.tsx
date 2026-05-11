import { createContext, useContext, useState, type ReactNode } from "react";

interface CredentialsContextValue {
  isReady: boolean;
  setReady: (ready: boolean) => void;
}

const CredentialsContext = createContext<CredentialsContextValue | null>(null);

export function CredentialsProvider({ children }: { children: ReactNode }) {
  const [isReady, setReady] = useState<boolean>(false);
  return (
    <CredentialsContext.Provider value={{ isReady, setReady }}>
      {children}
    </CredentialsContext.Provider>
  );
}

export function useCredentials(): CredentialsContextValue {
  const ctx = useContext(CredentialsContext);
  if (ctx === null) {
    throw new Error("useCredentials must be used within a CredentialsProvider");
  }
  return ctx;
}
