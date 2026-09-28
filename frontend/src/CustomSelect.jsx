import { Listbox, ListboxButton, ListboxOption, ListboxOptions } from "@headlessui/react";
import { Check, ChevronDown } from "lucide-react";

export default function CustomSelect({ value, options, onChange, ariaLabel }) {
  const normalizedOptions = options.map((option) => (
    typeof option === "string" ? { value: option, label: option, description: "" } : option
  ));
  const selectedOption = normalizedOptions.find((option) => option.value === value)
    ?? { value, label: value, description: "" };
  return (
    <Listbox value={value} onChange={onChange}>
      <div className="relative">
        <ListboxButton className="nyx-select-button" aria-label={ariaLabel}>
          <span className="min-w-0 truncate">{selectedOption.label}</span>
          <ChevronDown size={15} className="shrink-0 text-[var(--text-subtle)]" />
        </ListboxButton>
        <ListboxOptions anchor="bottom" transition className="nyx-select-options [--anchor-gap:6px] data-[closed]:-translate-y-[5px] data-[closed]:opacity-0">
          {normalizedOptions.map((option) => (
            <ListboxOption key={option.value} value={option.value} className="nyx-select-option group">
              <span className="min-w-0"><span className="block truncate font-medium">{option.label}</span>{option.description && <span className="mt-0.5 block truncate text-[10px] text-[var(--text-subtle)]">{option.description}</span>}</span>
              <Check size={14} className="shrink-0 opacity-0 group-data-[selected]:opacity-100" />
            </ListboxOption>
          ))}
        </ListboxOptions>
      </div>
    </Listbox>
  );
}
